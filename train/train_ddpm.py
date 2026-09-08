"""Stages 2 and 3 -- pretrain the 2D DDPM, then train the conditional DDPM.

``dataset.use_paired=false`` -> unconditional pretraining on motion-free slices
``dataset.use_paired=true``  -> conditional training on (target, corrupted) pairs
"""

import os

import hydra
import torch
from omegaconf import DictConfig, open_dict

from ddpm import GaussianDiffusion, Trainer, Unet2D
from ddpm.unet import UNet
from train.get_dataset import get_dataset


def _resolve_device(cfg):
    if torch.cuda.is_available():
        index = int(getattr(cfg.model, 'gpus', 0))
        torch.cuda.set_device(index)
        return torch.device(f'cuda:{index}')
    print('[warn] CUDA is unavailable -- falling back to CPU')
    return torch.device('cpu')


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    device = _resolve_device(cfg)

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name,
            cfg.model.results_folder_postfix)

    unet_dim = getattr(cfg.model, 'unet_dim', None) or cfg.model.diffusion_img_size

    if cfg.model.denoising_fn == 'Unet2D':
        model = Unet2D(
            dim=int(unet_dim),
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_mode=getattr(cfg.model, 'cond_mode', 'gated'),
            cond_gate_init=float(getattr(cfg.model, 'cond_gate_init', 0.1)),
            use_slice_pos_emb=bool(getattr(cfg.model, 'use_slice_pos_emb', True)),
        ).to(device)
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=2,
            use_slice_pos_emb=bool(getattr(cfg.model, 'use_slice_pos_emb', True)),
        ).to(device)
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    min_snr_gamma = getattr(cfg.model, 'min_snr_gamma', None)

    diffusion = GaussianDiffusion(
        model,
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

    train_dataset, *_ = get_dataset(cfg)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        max_grad_norm=getattr(cfg.model, 'max_grad_norm', None),
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone, map_location=device)

    trainer.train()


if __name__ == '__main__':
    run()
