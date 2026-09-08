@echo off
REM Stage 1 -- 2D VQ-GAN. Run from the repository root.
REM Edit DATA and OUT, then just run this file.

set DATA=D:/data/VQGAN_training
set OUT=D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D
set TAG=lf_dataset

set PYTHONPATH=%cd%
set PL_TORCH_DISTRIBUTED_BACKEND=gloo

python train\train_vqgan.py ^
  dataset=default dataset.root_dir="%DATA%" ^
  model=vq_gan_2d model.gpus=1 ^
  model.default_root_dir="%OUT%" model.default_root_dir_postfix=%TAG% ^
  model.precision=16 ^
  model.downsample=[4,4] model.embedding_dim=8 model.n_hiddens=64 ^
  model.num_groups=8 model.n_codes=8192 ^
  model.batch_size=16 model.num_workers=4 ^
  model.gradient_clip_val=1.0 model.lr=3e-4 ^
  model.discriminator_iter_start=20000 ^
  model.perceptual_weight=1 model.image_gan_weight=1 model.gan_feat_weight=4 ^
  model.background_weight=0.25 ^
  model.accumulate_grad_batches=1 model.max_epochs=50
