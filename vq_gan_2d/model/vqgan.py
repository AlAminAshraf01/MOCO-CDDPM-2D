"""2D VQ-GAN -- port of the 3D VQ-GAN used by MoCo-cDDPM.

Differences w.r.t. the 3D original (``vq_gan_3d/model/vqgan.py``):

* every ``Conv3d`` / ``ConvTranspose3d`` becomes its 2D counterpart and
  ``downsample`` is a 2-tuple ``[h, w]``;
* the *video* (3D) discriminator is removed -- in 2D there is only one
  discriminator, so ``video_gan_weight`` / ``NLayerDiscriminator3D`` are gone;
* the random-frame-selection trick (a 2D slice was drawn out of each volume to
  feed the image discriminator and LPIPS) disappears: the input *is* a slice,
  so the perceptual and adversarial losses see every pixel of every sample;
* low-field addition: an optional foreground-weighted reconstruction loss
  (``background_weight`` < 1) so the autoencoder does not spend its codebook on
  the Rician noise floor, which dominates the pixel count at 64 mT / 0.3 T.
"""

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

from vq_gan_2d.model.codebook import Codebook
from vq_gan_2d.model.lpips import LPIPS
from vq_gan_2d.utils import adopt_weight, shift_dim


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (torch.mean(F.softplus(-logits_real)) +
                  torch.mean(F.softplus(logits_fake)))


class VQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type,
                               cfg.model.padding_type, cfg.model.num_groups)
        self.decoder = Decoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type,
                               cfg.model.num_groups)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv2d(
            self.enc_out_ch, cfg.model.embedding_dim, 1,
            padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv2d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart,
                                 restart_thres=cfg.model.restart_thres)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels, cfg.model.disc_channels,
            cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss
        else:
            raise ValueError(
                f'Unknown disc_loss_type {cfg.model.disc_loss_type}')

        self.image_gan_weight = cfg.model.image_gan_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.l1_weight = cfg.model.l1_weight

        # Built lazily: constructing LPIPS downloads VGG16 + the LPIPS weights,
        # which is wasted work (and needs the network) when it is switched off.
        self.perceptual_model = LPIPS().eval() if self.perceptual_weight > 0 else None

        # --- low-field: down-weight the (Rician) background in the L1 term ---
        self.background_weight = float(
            getattr(cfg.model, 'background_weight', 1.0))
        self.foreground_threshold = float(
            getattr(cfg.model, 'foreground_threshold', -0.8))

        self.save_hyperparameters()

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    # ------------------------------------------------------------------ #
    # losses
    # ------------------------------------------------------------------ #
    def reconstruction_loss(self, x_recon, x):
        """L1, optionally weighted so background pixels count less.

        ``x`` is in [-1, 1]; at ultra-low field the air/noise region is by far
        the largest class, and an unweighted L1 lets the model trade anatomy
        for background fidelity.
        """
        if self.background_weight >= 1.0:
            return F.l1_loss(x_recon, x)
        with torch.no_grad():
            fg = (x > self.foreground_threshold).float()
            fg = F.max_pool2d(fg, kernel_size=5, stride=1, padding=2)
        w = self.background_weight + (1.0 - self.background_weight) * fg
        return (w * (x_recon - x).abs()).sum() / w.sum().clamp(min=1.0)

    def forward(self, x, optimizer_idx=None, log_image=False):
        # x: [B, C, H, W]
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = self.reconstruction_loss(x_recon, x) * self.l1_weight

        if log_image:
            return x, x_recon

        if optimizer_idx == 0:
            # ---- generator ----
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(
                    x, x_recon).mean() * self.perceptual_weight

            logits_image_fake, pred_image_fake = self.image_discriminator(
                x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_loss = self.image_gan_weight * g_image_loss
            disc_factor = adopt_weight(
                self.global_step,
                threshold=self.cfg.model.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            image_gan_feat_loss = 0
            feat_weights = 4.0 / (self.cfg.model.disc_layers + 1)
            if self.image_gan_weight > 0:
                _, pred_image_real = self.image_discriminator(x)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += feat_weights * F.l1_loss(
                        pred_image_fake[i], pred_image_real[i].detach())
            gan_feat_loss = disc_factor * self.gan_feat_weight * image_gan_feat_loss

            self.log("train/g_image_loss", g_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/image_gan_feat_loss", image_gan_feat_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/perceptual_loss", perceptual_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/commitment_loss", vq_output['commitment_loss'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log('train/perplexity', vq_output['perplexity'],
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss

        if optimizer_idx == 1:
            # ---- discriminator ----
            logits_image_real, _ = self.image_discriminator(x.detach())
            logits_image_fake, _ = self.image_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            disc_factor = adopt_weight(
                self.global_step,
                threshold=self.cfg.model.discriminator_iter_start)
            discloss = disc_factor * self.image_gan_weight * d_image_loss

            self.log("train/logits_image_real",
                     logits_image_real.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/logits_image_fake",
                     logits_image_fake.mean().detach(),
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/d_image_loss", d_image_loss,
                     logger=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            return discloss

        perceptual_loss = (self.perceptual_model(x, x_recon).mean()
                           * self.perceptual_weight
                           if self.perceptual_weight > 0
                           else torch.zeros((), device=x.device))
        return recon_loss, x_recon, vq_output, perceptual_loss

    # ------------------------------------------------------------------ #
    # lightning hooks
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['data']
        if optimizer_idx == 0:
            (recon_loss, _, vq_output, aeloss, perceptual_loss,
             gan_feat_loss) = self.forward(x, optimizer_idx)
            return (recon_loss + vq_output['commitment_loss'] + aeloss +
                    perceptual_loss + gan_feat_loss)
        return self.forward(x, optimizer_idx)

    def compute_ssim(self, preds, target):
        return structural_similarity_index_measure(
            preds.float(), target.float(), data_range=2.0)

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        recon_loss, x_recon, vq_output, perceptual_loss = self.forward(x)
        ssim = self.compute_ssim(x_recon, x)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss',
                 vq_output['commitment_loss'], prog_bar=True)
        self.log('val/ssim', ssim, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        capturable = bool(getattr(self.cfg.model, 'adam_capturable', False))
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9),
                                  capturable=capturable)
        opt_disc = torch.optim.Adam(self.image_discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9),
                                    capturable=capturable)
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        x = batch['data'].to(self.device)
        x, x_rec = self(x, log_image=True)
        return {"inputs": x, "reconstructions": x_rec}


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        num_groups = min(num_groups, in_channels)
        while num_groups > 1 and in_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
                            eps=1e-6, affine=True)
    return nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=1,
                 norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        assert len(n_times_downsample) == 2, \
            'downsample must be [h, w] for the 2D model'
        self.conv_blocks = nn.ModuleList()
        max_ds = int(n_times_downsample.max())

        self.conv_first = SamePadConv2d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        out_channels = n_hiddens
        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2 ** i
            out_channels = n_hiddens * 2 ** (i + 1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv2d(in_channels, out_channels, 4,
                                       stride=stride,
                                       padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels,
                                 norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups), SiLU())
        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        return self.final_block(h)


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group',
                 num_groups=32):
        super().__init__()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        assert len(n_times_upsample) == 2, \
            'downsample must be [h, w] for the 2D model'
        max_us = int(n_times_upsample.max())

        in_channels = n_hiddens * 2 ** max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups), SiLU())

        self.conv_blocks = nn.ModuleList()
        out_channels = in_channels
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens * \
                2 ** (max_us - i + 1)
            out_channels = n_hiddens * 2 ** (max_us - i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose2d(in_channels, out_channels, 4,
                                              stride=us)
            block.res1 = ResBlock(out_channels, out_channels,
                                  norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(out_channels, out_channels,
                                  norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv2d(out_channels, image_channel,
                                       kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for block in self.conv_blocks:
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        return self.conv_last(h)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, norm_type='group', padding_type='replicate',
                 num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv2d(in_channels, out_channels, kernel_size=3,
                                   padding_type=padding_type)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv2d(out_channels, out_channels, kernel_size=3,
                                   padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv2d(in_channels, out_channels,
                                               kernel_size=3,
                                               padding_type=padding_type)

    def forward(self, x):
        h = self.conv1(silu(self.norm1(x)))
        h = self.conv2(self.dropout(silu(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse: F.pad starts from the last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        self.pad_input = sum(pad_input, tuple())
        self.padding_type = padding_type

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        self.pad_input = sum(pad_input, tuple())
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias,
            padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 getIntermFeat=True):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2,
                               padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2,
                                    padding=padw),
                          norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1,
                                padding=padw),
                      norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1,
                                padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            stream = []
            for n in range(len(sequence)):
                stream += sequence[n]
            self.model = nn.Sequential(*stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                res.append(getattr(self, 'model' + str(n))(res[-1]))
            return res[-1], res[1:]
        return self.model(input), None
