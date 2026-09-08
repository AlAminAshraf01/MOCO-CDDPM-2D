"""MONAI-based 2D UNet -- the alternative denoiser (``denoising_fn=UNet``).

Port of ``ddpm/unet.py``.  The 3D file used ``(1, 3, 3)`` kernels and
``(1, 2, 2)`` strides to avoid downsampling the slice axis; in 2D those become
plain ``3`` and ``2``.  The gated spatial condition fusion at the first encoder
level is preserved, and the same optional slice-position embedding as
``Unet2D`` is available so the two denoisers stay interchangeable.
"""

import torch
import torch.nn as nn
from monai.networks.blocks import UnetBasicBlock, UnetOutBlock, UnetUpBlock
from monai.networks.layers.utils import get_act_layer

from ddpm.time_embedding import TimeEmbbeding


class DownBlock(nn.Module):
    def __init__(self, spatial_dims, in_ch, out_ch, time_emb_dim, cond_emb_dim,
                 act_name=("swish", {}), **kwargs):
        super().__init__()
        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, in_ch),
        )
        self.down_op = UnetBasicBlock(spatial_dims, in_ch, out_ch,
                                      act_name=act_name, **kwargs)

    def forward(self, x, time_emb, cond_emb):
        b, c, *_ = x.shape
        sp_dim = x.ndim - 2
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, c, *((1,) * sp_dim))
        return self.down_op(x + time_emb)


class UpBlock(nn.Module):
    def __init__(self, spatial_dims, skip_ch, enc_ch, time_emb_dim,
                 cond_emb_dim, act_name=("swish", {}), **kwargs):
        super().__init__()
        self.up_op = UnetUpBlock(spatial_dims, enc_ch, skip_ch,
                                 act_name=act_name, **kwargs)
        self.loca_time_embedder = nn.Sequential(
            get_act_layer(name=act_name),
            nn.Linear(time_emb_dim, enc_ch),
        )

    def forward(self, x_skip, x_enc, time_emb, cond_emb):
        b, c, *_ = x_enc.shape
        sp_dim = x_enc.ndim - 2
        time_emb = self.loca_time_embedder(time_emb)
        time_emb = time_emb.reshape(b, c, *((1,) * sp_dim))
        return self.up_op(x_enc + time_emb, x_skip)


class UNet(nn.Module):
    def __init__(self,
                 in_ch=1,
                 out_ch=1,
                 spatial_dims=2,
                 hid_chs=[32, 64, 128, 256, 512],
                 kernel_sizes=[3, 3, 3, 3, 3],
                 strides=[1, 2, 2, 2, 2],
                 upsample_kernel_sizes=None,
                 act_name=("SWISH", {}),
                 norm_name=("INSTANCE", {"affine": True}),
                 time_embedder=TimeEmbbeding,
                 time_embedder_kwargs={},
                 deep_ver_supervision=True,
                 estimate_variance=False,
                 use_self_conditioning=False,
                 cond_gate_strength=0.1,
                 use_slice_pos_emb=True,
                 **kwargs):
        super().__init__()
        assert spatial_dims == 2, 'this is the 2D port; use spatial_dims=2'
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = strides[1:]

        self.time_embedder = time_embedder(**time_embedder_kwargs)
        self.use_slice_pos_emb = use_slice_pos_emb
        if use_slice_pos_emb:
            self.slice_embedder = time_embedder(**time_embedder_kwargs)
        cond_emb_dim = None

        in_ch = in_ch * 2 if use_self_conditioning else in_ch
        self.inc = UnetBasicBlock(spatial_dims, in_ch, hid_chs[0],
                                  kernel_size=kernel_sizes[0],
                                  stride=strides[0], act_name=act_name,
                                  norm_name=norm_name, **kwargs)

        # ---- gated residual fusion (spatial) from the condition ----
        self.cond_to_hid1 = nn.Conv2d(in_ch, hid_chs[0], kernel_size=3,
                                      padding=1)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(hid_chs[0] * 2, hid_chs[0], kernel_size=1),
            get_act_layer(name=act_name),
        )
        self.fuse_norm1 = nn.InstanceNorm2d(hid_chs[0], affine=False)
        self.gate1_strength = cond_gate_strength

        self.encoders = nn.ModuleList([
            DownBlock(spatial_dims, hid_chs[i - 1], hid_chs[i],
                      time_emb_dim=self.time_embedder.emb_dim,
                      cond_emb_dim=cond_emb_dim, kernel_size=kernel_sizes[i],
                      stride=strides[i], act_name=act_name,
                      norm_name=norm_name, **kwargs)
            for i in range(1, len(strides))
        ])

        self.decoders = nn.ModuleList([
            UpBlock(spatial_dims, hid_chs[i], hid_chs[i + 1],
                    time_emb_dim=self.time_embedder.emb_dim,
                    cond_emb_dim=cond_emb_dim, kernel_size=kernel_sizes[i + 1],
                    stride=strides[i + 1], act_name=act_name,
                    norm_name=norm_name,
                    upsample_kernel_size=upsample_kernel_sizes[i], **kwargs)
            for i in range(len(strides) - 1)
        ])

        out_ch_hor = out_ch * 2 if estimate_variance else out_ch
        self.outc = UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor,
                                 dropout=None)
        if isinstance(deep_ver_supervision, bool):
            deep_ver_supervision = len(strides) - 2 if deep_ver_supervision else 0
        self.outc_ver = nn.ModuleList([
            UnetOutBlock(spatial_dims, hid_chs[i], out_ch, dropout=None)
            for i in range(1, deep_ver_supervision + 1)
        ])

    def forward(self, x_t, t, cond=None, slice_pos=None, self_cond=None,
                **kwargs):
        x = [None for _ in range(len(self.encoders) + 1)]
        x_t = torch.cat([x_t, self_cond], dim=1) if self_cond is not None else x_t
        x[0] = self.inc(x_t)

        if cond is not None and getattr(cond, 'ndim', None) == x_t.ndim:
            cond_feat = self.cond_to_hid1(cond)
            fused = self.fuse_conv1(torch.cat([x[0], cond_feat], dim=1))
            fused = torch.tanh(self.fuse_norm1(fused)) * 0.5
            x[0] = x[0] + self.gate1_strength * fused

        time_emb = self.time_embedder(t)
        if self.use_slice_pos_emb:
            if slice_pos is None:
                slice_pos = torch.full((x_t.shape[0],), 0.5,
                                       device=x_t.device)
            slice_pos = slice_pos.to(x_t.device).reshape(-1).float()
            time_emb = time_emb + self.slice_embedder(slice_pos * 1000.0)

        cond_emb = None
        for i in range(len(self.encoders)):
            x[i + 1] = self.encoders[i](x[i], time_emb, cond_emb)
        for i in range(len(self.decoders), 0, -1):
            x[i - 1] = self.decoders[i - 1](x[i - 1], x[i], time_emb, cond_emb)

        return self.outc(x[0])

    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        kwargs.pop('null_cond_prob', None)
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    model = UNet(in_ch=8, out_ch=8, spatial_dims=2)
    dummy = torch.randn((2, 8, 64, 64))
    time = torch.randint(0, 500, (2,)).float()
    print(model(dummy, time, cond=torch.randn_like(dummy)).shape)
