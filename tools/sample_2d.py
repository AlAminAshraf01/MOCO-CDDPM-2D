"""Slice-wise inference for the 2D conditional DDPM.

A 2D model has to denoise every slice of a volume separately, so this script
takes care of the whole round trip: volume -> slices -> latents -> samples ->
decoded slices -> volume, using exactly the preprocessing the dataset used.

Input modes (pick one):

* paired dataset   ``dataset.use_paired=true`` + ``dataset.root_dir``
* condition folder ``+model.condition_dir=...`` (``+model.target_dir=...`` optional)
* single file      ``+model.fixed_condition_path=...``

Example::

    set PYTHONPATH=%cd% && python tools\\sample_2d.py model=ddpm dataset=default ^
        model.vqgan_ckpt="...\\latest_checkpoint.ckpt" ^
        model.diffusion_img_size=64 model.diffusion_num_channels=8 ^
        model.dim_mults=[1,2,4,8] ^
        model.load_milestone="...\\model-120.pt" ^
        +model.condition_dir="D:\\data\\test\\corrupted" ^
        +model.target_dir="D:\\data\\test\\target" ^
        model.sampling_timesteps=250 +model.sample_batch_size=8
"""

import csv
import os
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from dataset.default import (ALL_EXTS, NIFTI_EXTS, Preprocess2D,
                             load_volume_as_slices, save_slices_as_volume)
from ddpm import GaussianDiffusion, Unet2D
from ddpm.diffusion import save_image_row
from ddpm.unet import UNet


# --------------------------------------------------------------------------- #
def build_model(cfg, device):
    unet_dim = getattr(cfg.model, 'unet_dim', None) or cfg.model.diffusion_img_size
    if cfg.model.denoising_fn == 'Unet2D':
        net = Unet2D(
            dim=int(unet_dim),
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_mode=getattr(cfg.model, 'cond_mode', 'gated'),
            cond_gate_init=float(getattr(cfg.model, 'cond_gate_init', 0.1)),
            use_slice_pos_emb=bool(getattr(cfg.model, 'use_slice_pos_emb', True)),
        ).to(device)
    elif cfg.model.denoising_fn == 'UNet':
        net = UNet(in_ch=cfg.model.diffusion_num_channels,
                   out_ch=cfg.model.diffusion_num_channels, spatial_dims=2,
                   use_slice_pos_emb=bool(
                       getattr(cfg.model, 'use_slice_pos_emb', True))).to(device)
    else:
        raise ValueError(f'Unknown denoising_fn: {cfg.model.denoising_fn}')

    min_snr_gamma = getattr(cfg.model, 'min_snr_gamma', None)
    diffusion = GaussianDiffusion(
        net,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        sampling_timesteps=getattr(cfg.model, 'sampling_timesteps', None),
        ddim_sampling_eta=float(getattr(cfg.model, 'ddim_sampling_eta', 0.0)),
        loss_type=cfg.model.loss_type,
        objective=getattr(cfg.model, 'objective', 'pred_noise'),
        beta_schedule=getattr(cfg.model, 'beta_schedule', 'cosine'),
        min_snr_gamma=None if min_snr_gamma in (None, 'null') else float(min_snr_gamma),
        cond_drop_prob=float(getattr(cfg.model, 'cond_drop_prob', 0.0)),
    ).to(device)
    return diffusion


def load_checkpoint(diffusion, path, device, use_ema=True):
    """Accepts Trainer checkpoints (``model``/``ema``) and raw state dicts."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and ('ema' in ckpt or 'model' in ckpt):
        key = 'ema' if (use_ema and 'ema' in ckpt) else 'model'
        state = ckpt[key]
        print(f'loading "{key}" weights from {path}')
    else:
        state = ckpt

    if any(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}

    if any(k.startswith('denoise_fn.') for k in state):
        result = diffusion.load_state_dict(state, strict=False)
    else:  # a bare U-Net state dict
        result = diffusion.denoise_fn.load_state_dict(state, strict=False)

    missing = [k for k in getattr(result, 'missing_keys', [])
               if not k.startswith('vqgan.')]
    unexpected = [k for k in getattr(result, 'unexpected_keys', [])
                  if not k.startswith('vqgan.')]
    if missing or unexpected:
        print(f'[warn] non-strict load: missing={missing[:8]}'
              f'{"..." if len(missing) > 8 else ""}, '
              f'unexpected={unexpected[:8]}'
              f'{"..." if len(unexpected) > 8 else ""}')
    diffusion.eval()
    return diffusion


# --------------------------------------------------------------------------- #
def _nii_names(directory):
    if not directory or not os.path.isdir(directory):
        return set()
    return {n for n in os.listdir(directory) if n.lower().endswith(NIFTI_EXTS)}


def build_pairs(cfg):
    """Return a list of ``(target_path_or_None, condition_path, name)``."""
    condition_dir = getattr(cfg.model, 'condition_dir', None)
    fixed_cond = getattr(cfg.model, 'fixed_condition_path', None)

    if bool(getattr(cfg.dataset, 'use_paired', False)) and not condition_dir \
            and not fixed_cond:
        tgt_dir = os.path.join(cfg.dataset.root_dir,
                               getattr(cfg.dataset, 'target_subdir', 'target'))
        cond_dir = os.path.join(cfg.dataset.root_dir,
                                getattr(cfg.dataset, 'condition_subdir',
                                        'corrupted'))
        names = sorted(_nii_names(tgt_dir) & _nii_names(cond_dir))
        return [(os.path.join(tgt_dir, n), os.path.join(cond_dir, n), n)
                for n in names]

    if condition_dir:
        target_dir = getattr(cfg.model, 'target_dir', None)
        names = sorted(_nii_names(condition_dir))
        target_names = _nii_names(target_dir)
        return [(os.path.join(target_dir, n) if n in target_names else None,
                 os.path.join(condition_dir, n), n) for n in names]

    if fixed_cond:
        return [(getattr(cfg.model, 'fixed_target_path', None), fixed_cond,
                 os.path.basename(fixed_cond))]

    raise RuntimeError('Provide dataset.use_paired=true, +model.condition_dir '
                       'or +model.fixed_condition_path')


def to01(tensor):
    return ((tensor.detach().float() + 1) / 2).clamp(0., 1.)


def slice_metrics(a, b, max_chunk=32):
    if a is None or b is None:
        return None, None
    a, b = to01(a), to01(b)
    ssim_vals, psnr_vals = [], []
    with torch.no_grad():
        for start in range(0, a.shape[0], max_chunk):
            end = min(start + max_chunk, a.shape[0])
            ssim_vals.append(structural_similarity_index_measure(
                b[start:end], a[start:end], data_range=1.0))
            psnr_vals.append(peak_signal_noise_ratio(
                b[start:end], a[start:end], data_range=1.0))
    return (torch.stack(ssim_vals).mean().item(),
            torch.stack(psnr_vals).mean().item())


@torch.inference_mode()
def sample_volume(diffusion, cond_slices, slice_pos, device, batch_size=8,
                  cond_scale=1.0):
    """Denoise a whole stack of slices in chunks."""
    outputs = []
    total = cond_slices.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = cond_slices[start:end].to(device)
        pos = slice_pos[start:end].to(device)
        latent_cond = diffusion.encode_condition(chunk)
        sample = diffusion.sample(cond=latent_cond, slice_pos=pos,
                                  cond_scale=cond_scale, progress=False)
        outputs.append(sample.detach().cpu())
        print(f'  slices {start}-{end - 1} / {total}')
    return torch.cat(outputs, dim=0)


# --------------------------------------------------------------------------- #
@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(int(getattr(cfg.model, 'gpus', 0)))

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name,
            cfg.model.results_folder_postfix)
    out_dir = getattr(cfg.model, 'output_dir', None) or os.path.join(
        cfg.model.results_folder, 'inference')
    os.makedirs(out_dir, exist_ok=True)

    pre = Preprocess2D.from_cfg(cfg.dataset)
    diffusion = build_model(cfg, device)
    if getattr(cfg.model, 'load_milestone', None):
        load_checkpoint(diffusion, cfg.model.load_milestone, device,
                        use_ema=bool(getattr(cfg.model, 'use_ema', True)))
    else:
        print('[warn] no model.load_milestone given -- sampling from an '
              'untrained network')

    pairs = build_pairs(cfg)
    max_files = getattr(cfg.model, 'max_files', None)
    if max_files:
        pairs = pairs[:int(max_files)]
    print(f'Found {len(pairs)} volume(s) to process')

    seed = getattr(cfg.model, 'fixed_sampling_seed', None)
    batch_size = int(getattr(cfg.model, 'sample_batch_size', 8))
    cond_scale = float(getattr(cfg.model, 'cond_scale', 1.0))
    save_nifti = bool(getattr(cfg.model, 'save_nifti', True))
    save_png = bool(getattr(cfg.model, 'save_png', True))

    csv_path = os.path.join(out_dir, 'sampling_metrics.csv')
    write_header = not os.path.exists(csv_path)
    fields = ['filename', 'timestamp', 'n_slices',
              'ssim_target_sample', 'psnr_target_sample',
              'ssim_target_condition', 'psnr_target_condition']

    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()

        for index, (target_path, cond_path, name) in enumerate(pairs):
            print(f'[{index + 1}/{len(pairs)}] {name}')
            if seed is not None:
                torch.manual_seed(int(seed) + index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed) + index)

            cond_slices, meta = load_volume_as_slices(cond_path, pre)
            samples = sample_volume(diffusion, cond_slices, meta['slice_pos'],
                                    device, batch_size=batch_size,
                                    cond_scale=cond_scale)

            target_slices = None
            if target_path and os.path.exists(target_path):
                target_slices, _ = load_volume_as_slices(target_path, pre)

            base = name
            for ext in NIFTI_EXTS:
                if base.lower().endswith(ext):
                    base = base[:-len(ext)]
                    break

            if save_nifti:
                save_slices_as_volume(
                    samples, meta, os.path.join(out_dir, f'{base}_sample.nii.gz'))
            if save_png:
                step = max(1, samples.shape[0] // 8)
                picks = list(range(0, samples.shape[0], step))[:8]
                save_image_row(
                    [cond_slices[picks], samples[picks],
                     target_slices[picks] if target_slices is not None else None],
                    os.path.join(out_dir, f'{base}_comparison.png'),
                    max_items=len(picks))

            ssim_ts, psnr_ts = slice_metrics(target_slices, samples)
            ssim_tc, psnr_tc = slice_metrics(target_slices, cond_slices)
            print(f'  SSIM target/sample={ssim_ts}  target/condition={ssim_tc}')

            writer.writerow({
                'filename': name,
                'timestamp': datetime.now().isoformat(),
                'n_slices': int(samples.shape[0]),
                'ssim_target_sample': ssim_ts,
                'psnr_target_sample': psnr_ts,
                'ssim_target_condition': ssim_tc,
                'psnr_target_condition': psnr_tc,
            })
            handle.flush()

    print(f'done -- outputs in {out_dir}')


if __name__ == '__main__':
    main()
