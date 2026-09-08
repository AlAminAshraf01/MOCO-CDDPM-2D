@echo off
REM Stage 3 -- conditional DDPM on paired (target, corrupted) slices.
REM MILESTONE should point at the stage-2 checkpoint you want to start from.

set DATA=D:/data/cddpm_2d
set VQGAN=D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D/DEFAULT/lf_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt
set OUT=D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d
set TAG=cddpm_lf
set MILESTONE=D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d/DEFAULT/pretrain_lf/model-120.pt

set PYTHONPATH=%cd%

python train\train_ddpm.py ^
  model=ddpm dataset=default ^
  dataset.use_paired=true dataset.root_dir="%DATA%" ^
  dataset.target_subdir=target dataset.condition_subdir=corrupted ^
  model.denoising_fn=Unet2D model.vqgan_ckpt="%VQGAN%" ^
  model.diffusion_img_size=64 model.diffusion_num_channels=8 ^
  model.dim_mults=[1,2,4,8] ^
  model.batch_size=16 model.num_workers=4 ^
  model.gradient_accumulate_every=1 model.train_lr=1e-4 ^
  model.ema_decay=0.995 model.save_and_sample_every=1000 ^
  model.gpus=0 model.amp=True +model.fixed_sampling_seed=1 ^
  model.results_folder="%OUT%" model.results_folder_postfix=%TAG% ^
  model.load_milestone="%MILESTONE%"
