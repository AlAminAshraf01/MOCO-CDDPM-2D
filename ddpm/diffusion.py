"""2D latent conditional DDPM -- port of ``ddpm/diffusion.py`` from MoCo-cDDPM.

What changed going from 3D to 2D
--------------------------------
* ``Unet3D`` -> ``Unet2D``.  All ``(1, k, k)`` pseudo-2D convolutions become
  real ``k x k`` convolutions, and the whole *temporal* branch is gone:
  ``RelativePositionBias``, ``RotaryEmbedding``, the per-block temporal
  attention and ``focus_present_mask`` had no meaning once the depth axis is
  removed.  Sparse spatial linear attention and the mid-block full attention
  are kept.
* The gated residual condition fusion -- the part that makes this model
  *conditional* on the motion-corrupted image -- is preserved exactly, in 2D.
  ``cond_mode='concat'`` is offered as a stronger alternative.
* Slice-position embedding (``use_slice_pos_emb``) is added.  It replaces, at
  almost no cost, the only genuinely useful thing the temporal attention gave
  the 3D model: knowing where in the stack the current slice sits.
* DDIM sampling is implemented (it was configured but unused in the 3D repo).
  A 2D model has to denoise every slice separately, so per-volume inference is
  ``n_slices x T`` network evaluations -- 250-step DDIM instead of 500-step
  ancestral sampling is what makes that practical.
* ``objective`` is now honoured.  The 3D config said ``pred_x0`` but the code
  always trained epsilon-prediction; the default here is ``pred_noise``, i.e.
  what the 3D model actually did.

Low-field specific knobs
------------------------
* ``min_snr_gamma``: Min-SNR-gamma loss weighting.  Low-field cohorts are small
  and this materially stabilises training on a few thousand slices.
* ``cond_drop_prob`` > 0 trains a null condition so that classifier-free
  guidance (``cond_scale`` > 1) becomes usable at sampling time.
* ``loss_type='l1'`` is kept as the default: it is the robust choice against
  the heavy-tailed residuals a Rician noise floor produces.
"""

import copy
import csv
import math
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from tqdm import tqdm

from vq_gan_2d.model.vqgan import VQGAN

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def safe_groups(groups, channels):
    """Largest power-of-two-ish group count that divides ``channels``."""
    groups = min(groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def as_pair(value):
    if isinstance(value, (tuple, list)):
        assert len(value) == 2
        return int(value[0]), int(value[1])
    return int(value), int(value)


class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(),
                                             ma_model.parameters()):
            if not current_params.requires_grad:
                continue
            ma_params.data = self.update_average(ma_params.data,
                                                 current_params.data)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(safe_groups(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    """Same module as the 3D version, minus the frame folding."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        return rearrange(x, f'{self.to_einops} -> {self.from_einops}',
                         **reconstitute_kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
        q = q * self.scale

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# --------------------------------------------------------------------------- #
# denoising network
# --------------------------------------------------------------------------- #
class Unet2D(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        resnet_groups=8,
        cond_mode='gated',            # 'gated' | 'concat' | 'both'
        cond_gate_init=0.1,
        learn_cond_gate=True,
        use_slice_pos_emb=True,
    ):
        super().__init__()
        self.channels = channels
        assert cond_mode in ('gated', 'concat', 'both')
        self.cond_mode = cond_mode
        self.use_concat_cond = cond_mode in ('concat', 'both')
        self.use_gated_cond = cond_mode in ('gated', 'both')
        self.use_slice_pos_emb = use_slice_pos_emb

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2

        in_channels = channels * 2 if self.use_concat_cond else channels
        self.init_conv = nn.Conv2d(in_channels, init_dim, init_kernel_size,
                                   padding=init_padding)

        # learned null condition -> makes classifier-free guidance possible
        self.null_cond = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- gated residual fusion of the spatial condition (as in the 3D model) ---
        if self.use_gated_cond:
            self.cond_to_init = nn.Conv2d(channels, init_dim, 3, padding=1)
            self.fuse_init = nn.Conv2d(init_dim * 2, init_dim, 1)
            self.fuse_init_act = nn.SiLU()
            self.fuse_init_norm = nn.InstanceNorm2d(init_dim, affine=False)
            gate_logit = math.log(cond_gate_init / (1. - cond_gate_init))
            self.fuse_gate_raw = nn.Parameter(
                torch.tensor(gate_logit, dtype=torch.float32),
                requires_grad=bool(learn_cond_gate))

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        if use_slice_pos_emb:
            self.slice_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        cond_dim = time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(
                    dim_out, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom(
            'b c h w', 'b (h w) c', Attention(mid_dim, heads=attn_heads))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(
                    dim_in, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv2d(dim, out_dim, 1),
        )

    # ------------------------------------------------------------------ #
    def _resolve_cond(self, x, cond, null_cond_prob):
        """Return the condition to actually use, applying condition dropout."""
        batch, device = x.shape[0], x.device
        null = self.null_cond.expand(batch, -1, x.shape[-2], x.shape[-1])

        if not exists(cond):
            # unconditional pretraining: behave exactly like the 3D model and
            # skip fusion entirely, unless concat needs a tensor.
            return null if (self.use_concat_cond or null_cond_prob >= 1.0) else None

        null = null.to(cond.dtype)      # autocast may hand us fp16 latents
        if null_cond_prob > 0:
            mask = prob_mask_like((batch,), null_cond_prob, device)
            cond = torch.where(rearrange(mask, 'b -> b 1 1 1'), null, cond)
        return cond

    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, cond=None, slice_pos=None, null_cond_prob=0.,
                **unused):
        cond = self._resolve_cond(x, cond, null_cond_prob)

        if self.use_concat_cond:
            x = torch.cat((x, cond), dim=1)

        x = self.init_conv(x)

        if self.use_gated_cond and exists(cond):
            cond_feat = self.cond_to_init(cond)
            fused = self.fuse_init(torch.cat([x, cond_feat], dim=1))
            fused = self.fuse_init_act(fused)
            fused = torch.tanh(self.fuse_init_norm(fused)) * 0.5
            gate = torch.sigmoid(self.fuse_gate_raw)
            x = x + gate * fused

        r = x.clone()

        t = self.time_mlp(time)
        if self.use_slice_pos_emb:
            if not exists(slice_pos):
                slice_pos = torch.full((x.shape[0],), 0.5, device=x.device,
                                       dtype=t.dtype)
            slice_pos = slice_pos.to(device=x.device, dtype=t.dtype).reshape(-1)
            # scale into the same numeric range the timestep embedding sees
            t = t + self.slice_mlp(slice_pos * 1000.0)

        h = []
        for block1, block2, spatial_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# --------------------------------------------------------------------------- #
# gaussian diffusion
# --------------------------------------------------------------------------- #
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 1e-4, scale * 0.02, timesteps,
                          dtype=torch.float64)


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=500,
        sampling_timesteps=None,
        ddim_sampling_eta=0.,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        min_snr_gamma=None,
        cond_drop_prob=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = as_pair(image_size)
        self.denoise_fn = denoise_fn
        self.objective = objective
        assert objective in ('pred_noise', 'pred_x0')
        self.cond_drop_prob = cond_drop_prob

        if vqgan_ckpt:
            self.vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
            self.vqgan.eval()
            for param in self.vqgan.parameters():
                param.requires_grad_(False)
            self.register_buffer('_vq_embedding_min',
                                 self.vqgan.codebook.embeddings.min().clone())
            self.register_buffer('_vq_embedding_max',
                                 self.vqgan.codebook.embeddings.max().clone())
        else:
            self.vqgan = None
            self.register_buffer('_vq_embedding_min', torch.tensor(0.))
            self.register_buffer('_vq_embedding_max', torch.tensor(1.))

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta_schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = int(default(sampling_timesteps, timesteps))
        assert self.sampling_timesteps <= self.num_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.loss_type = loss_type

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1',
                        betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2',
                        (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Min-SNR-gamma loss weighting (Hang et al., 2023)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if min_snr_gamma is not None:
            clipped = snr.clone().clamp(max=float(min_snr_gamma))
            weight = clipped / snr if objective == 'pred_noise' else clipped
        else:
            weight = torch.ones_like(snr)
        register_buffer('loss_weight', weight)

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def train(self, mode=True):
        """Keep the frozen VQ-GAN in eval mode, always.

        ``Codebook.forward`` performs EMA updates and random restarts whenever
        ``self.training`` is True.  ``decode(..., quantize=True)`` runs the
        codebook, so a stray ``.train()`` would let *generated* latents rewrite
        the autoencoder's codebook mid-training.
        """
        super().train(mode)
        if isinstance(self.vqgan, VQGAN):
            self.vqgan.eval()
        return self

    # ------------------------------------------------------------------ #
    # VQ-GAN latent space
    # ------------------------------------------------------------------ #
    def _normalize_vq_embeddings(self, embeddings):
        denom = self._vq_embedding_max - self._vq_embedding_min
        return ((embeddings - self._vq_embedding_min) / denom) * 2.0 - 1.0

    def _denormalize_vq_embeddings(self, latent):
        denom = self._vq_embedding_max - self._vq_embedding_min
        return ((latent + 1.0) / 2.0) * denom + self._vq_embedding_min

    def _encode_with_vqgan(self, x):
        with torch.no_grad():
            embeddings = self.vqgan.encode(x, quantize=False,
                                           include_embeddings=True)
        return self._normalize_vq_embeddings(embeddings)

    def _is_latent(self, tensor):
        return (tensor.shape[1] == self.channels
                and tuple(tensor.shape[-2:]) == self.image_size)

    def encode_condition(self, cond):
        if not exists(cond):
            return cond
        if isinstance(self.vqgan, VQGAN):
            if not self._is_latent(cond):
                cond = self._encode_with_vqgan(cond)
        return cond

    def decode_latent(self, latent):
        if not isinstance(self.vqgan, VQGAN):
            return latent
        return self.vqgan.decode(self._denormalize_vq_embeddings(latent),
                                 quantize=True)

    # ------------------------------------------------------------------ #
    # q / p
    # ------------------------------------------------------------------ #
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _maybe_clip(self, x_recon, clip_denoised):
        if not clip_denoised:
            return x_recon
        s = 1.
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x_recon, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile, dim=-1)
            s.clamp_(min=1.)
            s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
        return x_recon.clamp(-s, s) / s

    def model_predictions(self, x, t, cond=None, slice_pos=None, cond_scale=1.,
                          clip_denoised=True):
        model_out = self.denoise_fn.forward_with_cond_scale(
            x, t, cond=cond, slice_pos=slice_pos, cond_scale=cond_scale)

        if self.objective == 'pred_noise':
            pred_noise = model_out
            x_start = self._maybe_clip(
                self.predict_start_from_noise(x, t, pred_noise), clip_denoised)
            if clip_denoised:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            x_start = self._maybe_clip(model_out, clip_denoised)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None,
                        slice_pos=None, cond_scale=1.):
        _, x_start = self.model_predictions(
            x, t, cond=cond, slice_pos=slice_pos, cond_scale=cond_scale,
            clip_denoised=clip_denoised)
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, slice_pos=None, cond_scale=1.,
                 clip_denoised=True):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond,
            slice_pos=slice_pos, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, slice_pos=None, cond_scale=1.,
                      progress=True):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        steps = reversed(range(0, self.num_timesteps))
        if progress:
            steps = tqdm(steps, desc='ddpm sampling', total=self.num_timesteps)
        for i in steps:
            img = self.p_sample(img,
                                torch.full((b,), i, device=device, dtype=torch.long),
                                cond=cond, slice_pos=slice_pos,
                                cond_scale=cond_scale)
        return img

    @torch.inference_mode()
    def ddim_sample_loop(self, shape, cond=None, slice_pos=None, cond_scale=1.,
                         clip_denoised=True, progress=True):
        device = self.betas.device
        b = shape[0]
        total, sampling = self.num_timesteps, self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total - 1, steps=sampling + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        iterator = tqdm(time_pairs, desc='ddim sampling') if progress else time_pairs

        for time, time_next in iterator:
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                img, t, cond=cond, slice_pos=slice_pos, cond_scale=cond_scale,
                clip_denoised=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            if eta > 0:
                # clamp before sqrt: the product can dip just below zero from
                # float error, and eta * nan would still be nan
                sigma = eta * (((1 - alpha / alpha_next) * (1 - alpha_next) /
                                (1 - alpha)).clamp(min=0)).sqrt()
            else:
                sigma = torch.zeros((), device=device)
            c = (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt()

            noise = torch.randn_like(img) if float(sigma) > 0 else 0.
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return img

    @torch.inference_mode()
    def sample(self, cond=None, slice_pos=None, cond_scale=1., batch_size=16,
               use_ddim=None, decode=True, progress=True):
        device = next(self.denoise_fn.parameters()).device

        if exists(cond):
            cond = cond.to(device)
            cond = self.encode_condition(cond)
            batch_size = cond.shape[0]
        if exists(slice_pos):
            slice_pos = torch.as_tensor(slice_pos).to(device).reshape(-1)

        shape = (batch_size, self.channels, *self.image_size)
        use_ddim = default(use_ddim, self.is_ddim_sampling)
        loop = self.ddim_sample_loop if use_ddim else self.p_sample_loop
        latent = loop(shape, cond=cond, slice_pos=slice_pos,
                      cond_scale=cond_scale, progress=progress)

        if not decode:
            return latent
        if isinstance(self.vqgan, VQGAN):
            return self.decode_latent(latent)
        return latent

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def p_losses(self, x_start, t, cond=None, slice_pos=None, noise=None,
                 **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond = self.encode_condition(cond)
        model_out = self.denoise_fn(x_noisy, t, cond=cond, slice_pos=slice_pos,
                                    null_cond_prob=self.cond_drop_prob, **kwargs)

        target = noise if self.objective == 'pred_noise' else x_start

        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_out, target, reduction='none')
        else:
            raise NotImplementedError(self.loss_type)

        loss = rearrange(loss, 'b ... -> b (...)').mean(dim=1)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x, cond=None, slice_pos=None, **kwargs):
        if isinstance(self.vqgan, VQGAN):
            x = self._encode_with_vqgan(x)

        b, c, h, w = x.shape
        assert c == self.channels and (h, w) == self.image_size, (
            f'latent shape {(c, h, w)} does not match the configured '
            f'{(self.channels, *self.image_size)} -- check '
            f'diffusion_num_channels / diffusion_img_size against the VQ-GAN')

        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        return self.p_losses(x, t, cond=cond, slice_pos=slice_pos, **kwargs)


# --------------------------------------------------------------------------- #
# visualisation helpers (PNG grids replace the 3D GIFs)
# --------------------------------------------------------------------------- #
def _to_uint8(tensor):
    tensor = tensor.detach().float().cpu()
    tensor = ((tensor + 1.0) / 2.0).clamp(0., 1.)
    return (tensor.numpy() * 255).astype(np.uint8)


def save_image_row(tensors, path, labels=None, max_items=8):
    """Stack ``[B, 1, H, W]`` tensors: one column per tensor, one row per item."""
    import imageio.v2 as imageio

    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    n = min(max_items, min(t.shape[0] for t in tensors))
    arrays = [_to_uint8(t[:n]) for t in tensors]

    h, w = arrays[0].shape[-2:]
    pad = 2
    rows = []
    for i in range(n):
        cols = [np.pad(a[i, 0], pad, constant_values=0) for a in arrays]
        rows.append(np.concatenate(cols, axis=1))
    grid = np.concatenate(rows, axis=0)
    imageio.imwrite(path, grid)
    return path


def save_single_image(tensor, path):
    import imageio.v2 as imageio
    imageio.imwrite(path, _to_uint8(tensor)[0, 0])
    return path


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(
        self,
        diffusion_model,
        cfg,
        dataset=None,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        num_sample_rows=1,
        max_grad_norm=None,
        num_workers=4,
    ):
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        # the frozen VQ-GAN is identical in both copies -- share it
        self.ema_model.vqgan = self.model.vqgan
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.cfg = cfg
        assert dataset is not None, 'Provide a dataset'
        self.ds = dataset
        assert len(self.ds) > 0, 'the dataset is empty'
        print(f'found {len(self.ds)} 2D slices')

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True,
                        pin_memory=True, num_workers=num_workers,
                        drop_last=True)
        self.len_dataloader = len(dl)
        self.dl = cycle(dl)

        trainable = [p for p in diffusion_model.parameters() if p.requires_grad]
        self.opt = Adam(trainable, lr=train_lr)

        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.is_conditioned = False
        self._last_condition = None
        self._last_condition_data = None
        self._last_target_data = None
        self._last_slice_pos = None

        self.reset_parameters()

    # ------------------------------------------------------------------ #
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict(), strict=False)

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    @staticmethod
    def _strip_vqgan(state_dict):
        return {k: v for k, v in state_dict.items() if not k.startswith('vqgan.')}

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self._strip_vqgan(self.model.state_dict()),
            'ema': self._strip_vqgan(self.ema_model.state_dict()),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, map_location=None, strict=False, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'no milestone to load from'
            milestone = str(self.results_folder / f'model-{max(all_milestones)}.pt')

        data = (torch.load(milestone, map_location=map_location)
                if map_location else torch.load(milestone))

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=strict)
        self.ema_model.load_state_dict(data['ema'], strict=strict)
        self.scaler.load_state_dict(data['scaler'])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _expand(tensor, total, device):
        if tensor is None:
            return None
        tensor = tensor.to(device)
        if tensor.shape[0] < total:
            repeat = (total + tensor.shape[0] - 1) // tensor.shape[0]
            tensor = tensor.repeat(repeat, *([1] * (tensor.ndim - 1)))
        return tensor[:total]

    @staticmethod
    def _metrics(a, b, max_chunk=64):
        """SSIM / PSNR on [0, 1] image-space stacks ``[N, C, H, W]``."""
        if a is None or b is None:
            return None, None
        n = a.shape[0]
        ssim_vals, psnr_vals = [], []
        with torch.no_grad():
            for start in range(0, n, max_chunk):
                end = min(start + max_chunk, n)
                ssim_vals.append(structural_similarity_index_measure(
                    b[start:end], a[start:end], data_range=1.0))
                psnr_vals.append(peak_signal_noise_ratio(
                    b[start:end], a[start:end], data_range=1.0))
        return (torch.stack(ssim_vals).mean().item(),
                torch.stack(psnr_vals).mean().item())

    def train(self, log_fn=noop):
        assert callable(log_fn)
        device = next(self.model.parameters()).device

        while self.step < self.train_num_steps:
            last_condition = None
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                data = batch['data'].to(device)
                self._last_target_data = data.detach()

                slice_pos = batch.get('slice_pos')
                if slice_pos is not None:
                    slice_pos = slice_pos.to(device)
                self._last_slice_pos = slice_pos

                cond = batch.get('cond')
                if cond is not None:
                    cond = cond.to(device)
                    self._last_condition_data = cond.detach()
                    cond_latent = self.model.encode_condition(cond)
                    last_condition = cond_latent.detach()
                    self.is_conditioned = True
                else:
                    self._last_condition_data = None
                    cond_latent = None

                with autocast(enabled=self.amp):
                    loss = self.model(data, cond=cond_latent,
                                      slice_pos=slice_pos)
                    self.scaler.scale(
                        loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            if last_condition is not None:
                self._last_condition = last_condition

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self._sample_and_log(log, device)
                self.save(self.step // self.save_and_sample_every)

            log_fn(log)
            self.step += 1

        print('training completed')

    # ------------------------------------------------------------------ #
    def _sample_and_log(self, log, device):
        self.ema_model.eval()
        milestone = self.step // self.save_and_sample_every
        # how many slices to denoise for the milestone preview/metrics. The 3D
        # repo used num_sample_rows**2 (= 1); a handful gives far less noisy
        # SSIM/PSNR without costing much, since 2D sampling is cheap.
        num_samples = int(getattr(self.cfg.model, 'num_eval_samples',
                                  max(4, self.num_sample_rows ** 2)))
        num_samples = max(1, min(num_samples, self.batch_size))

        fixed_seed = getattr(self.cfg.model, 'fixed_sampling_seed', None)
        if fixed_seed is not None:
            torch.manual_seed(int(fixed_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(fixed_seed))

        with torch.no_grad():
            slice_pos = self._expand(self._last_slice_pos, num_samples, device)
            if self.is_conditioned:
                cond = self._expand(self._last_condition, num_samples, device)
                samples = self.ema_model.sample(
                    cond=cond, slice_pos=slice_pos,
                    cond_scale=float(getattr(self.cfg.model, 'cond_scale', 1.0)),
                    progress=False)
            else:
                samples = self.ema_model.sample(batch_size=num_samples,
                                                slice_pos=slice_pos,
                                                progress=False)

        target = self._expand(self._last_target_data, num_samples, device)
        condition = self._expand(self._last_condition_data, num_samples, device)
        if target is not None:
            target = target.type_as(samples)
        if condition is not None:
            condition = condition.type_as(samples)

        def to01(t):
            if t is None:
                return None
            return ((t.detach().float() + 1) / 2).clamp(0., 1.)

        ssim_ts, psnr_ts = self._metrics(to01(target), to01(samples))
        ssim_tc, psnr_tc = self._metrics(to01(target), to01(condition))

        metrics_path = self.results_folder / 'sampling_metrics.csv'
        fieldnames = ['step', 'milestone', 'timestamp',
                      'ssim_target_sample', 'psnr_target_sample',
                      'ssim_target_condition', 'psnr_target_condition']
        write_header = not metrics_path.exists()
        with metrics_path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'step': self.step, 'milestone': milestone,
                'timestamp': datetime.now().isoformat(),
                'ssim_target_sample': ssim_ts,
                'psnr_target_sample': psnr_ts,
                'ssim_target_condition': ssim_tc,
                'psnr_target_condition': psnr_tc,
            })

        comparison = save_image_row(
            [condition, samples, target],
            str(self.results_folder / f'{milestone}_comparison.png'))
        save_image_row([samples],
                       str(self.results_folder / f'sample-{milestone}.png'))
        log.update({'sample': comparison,
                    'ssim': ssim_ts, 'psnr': psnr_ts})
        # deliberately left in eval(): the EMA model is only ever sampled from,
        # and putting it back in train mode would also wake the shared VQ-GAN.
