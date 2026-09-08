@echo off
REM Slice-wise inference over a folder of motion-corrupted volumes.

set VQGAN=D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D/DEFAULT/lf_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt
set MILESTONE=D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d/DEFAULT/cddpm_lf/model-300.pt
set COND=D:/data/test/corrupted
set TARGET=D:/data/test/target
set OUTDIR=D:/out/inference

set PYTHONPATH=%cd%

python tools\sample_2d.py ^
  model=ddpm dataset=default ^
  model.vqgan_ckpt="%VQGAN%" ^
  model.diffusion_img_size=64 model.diffusion_num_channels=8 ^
  model.dim_mults=[1,2,4,8] ^
  model.load_milestone="%MILESTONE%" ^
  model.sampling_timesteps=250 model.cond_scale=1.0 ^
  +model.condition_dir="%COND%" +model.target_dir="%TARGET%" ^
  +model.sample_batch_size=8 +model.output_dir="%OUTDIR%" ^
  +model.fixed_sampling_seed=1
