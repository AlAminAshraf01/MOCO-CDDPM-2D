# MoCo-cDDPM-2D

A 2D port of [MoCo-cDDPM](https://github.com/alfattah129/MOCO-CDDPM), retuned for
**low-field / ultra-low-field MRI** (64 mT Hyperfine, 0.3 T M4Raw) instead of 3 T.

Same three-stage workflow as the 3D original:

1. Train a **2D VQ-GAN** to compress slices into a latent representation.
2. Pretrain a **2D DDPM** on motion-free slices.
3. Train a **conditional 2D DDPM** on paired motion-corrupted / motion-free slices.

---

## Why 2D is the right model for this data

Not just a convenience — for the datasets in question it is the more faithful choice:

* **The artefact is in-plane.** In a 2D multi-slice sequence (which is what M4Raw and
  most 64 mT protocols use) phase encoding happens *within* the slice, so rigid head
  motion imprints ghosting and blurring along the in-plane phase-encode direction of
  that slice. A 2D network sees the artefact in the plane where it actually lives; a 3D
  network spends capacity modelling a through-plane correlation the acquisition does
  not produce.
* **Through-plane resolution is poor anyway.** At 64 mT slices are typically ~5 mm
  thick against ~1.5 mm in plane. The 3D model's depth axis carried little real
  anatomical continuity.
* **Sample count.** Low-field cohorts are small. 100 volumes × 18 slices is 1 800
  training samples instead of 100 — the difference between trainable and not.
* **Cost.** A 256×256 slice is roughly two orders of magnitude cheaper than a
  160×160×64 volume, so the VQ-GAN can be *wider* (`n_hiddens` 64 instead of 16),
  which matters when you cannot afford to lose SNR in the latent bottleneck.

The cost of going 2D is the loss of through-plane context. That is recovered cheaply by
a **slice-position embedding** (`use_slice_pos_emb`), which tells the denoiser where in
the stack the current slice sits.

---

## What changed, file by file

| 3D | 2D | Change |
|---|---|---|
| `vq_gan_3d/` | `vq_gan_2d/` | `Conv3d`→`Conv2d`, `downsample` is `[h, w]`. The **3D "video" discriminator is removed** (only one discriminator exists in 2D), as is the random-frame-selection trick that picked one slice per volume for LPIPS/GAN — the input *is* a slice now, so those losses see every pixel. |
| `ddpm/diffusion.py` `Unet3D` | `Unet2D` | Whole temporal branch deleted: `RelativePositionBias`, `RotaryEmbedding`, per-block temporal attention, `focus_present_mask`. `(1,k,k)` pseudo-2D convs become real `k×k`. Spatial linear attention and mid-block full attention kept. **Gated residual condition fusion kept exactly** (the model's core contribution), plus `cond_mode='concat'` as an alternative. |
| `GaussianDiffusion` | `GaussianDiffusion` | `num_frames` gone; shapes are `b c h w`. **DDIM sampling implemented** (configured but unused in the 3D repo). `objective` now honoured. Min-SNR-γ weighting and classifier-free guidance added. |
| `Trainer` | `Trainer` | GIF logging → PNG comparison grids (condition ∥ sample ∥ target). The frozen VQ-GAN is shared with the EMA copy and stripped from checkpoints. |
| `dataset/default.py` | rewritten | torchio volume pipeline → nibabel slice-level pipeline with percentile scaling, `-1` padding, empty-slice rejection, and paired-safe augmentation. |
| `ddpm/unet.py` | `spatial_dims=2` | MONAI alternative denoiser, `(1,3,3)`/`(1,2,2)` → `3`/`2`. |
| `evaluation/*.ipynb`, `tools/sample_once.py` | `tools/sample_2d.py`, `evaluation/evaluate_2d.py`, `tools/test_vqgan_2d.py` | Scripts instead of notebooks; whole-volume slice-wise inference. |
| — | `tools/simulate_motion_2d.py` | **New.** 2D in-plane rigid-motion simulation in k-space, to build the paired data a 2D model needs. |
| `comparison models/*.ipynb` | not ported | The 3D baselines (UNet3D, SwinUNETR, HighResNet, DynUNet) are separate work; MONAI provides 2D variants of all four via `spatial_dims=2`. |

### Two things the 3D code got wrong, fixed here

* `tio.CropOrPad` ran **after** `RescaleIntensity(-1, 1)`, so padding used `0` — mid-grey.
  The 2D pipeline pads with `-1` (background).
* `config/model/ddpm.yaml` declared `objective: pred_x0`, but `p_losses` always
  computed `F.l1_loss(noise, x_recon)`, i.e. ε-prediction. `objective` is now actually
  wired up; the default `pred_noise` reproduces what the 3D model really did.

---

## Low-field adjustments

The 3D model was tuned on 3 T data. These are the changes that matter at 64 mT / 0.3 T,
with the reasoning — all are config values, so you can revert any of them.

| Setting | 3T value | LF value | Why |
|---|---|---|---|
| `dataset.percentiles` | min/max | `[0.5, 99.5]` | At low SNR one hot voxel sets the max and squashes the brain into a fraction of `[-1, 1]`. |
| `dataset.min_foreground_fraction` | — | `0.02` | End slices of a low-field stack are pure Rician noise; training on them teaches the model to generate noise. |
| `dataset.pad_value` | `0` | `-1.0` | See above. |
| `model.perceptual_weight` (VQ-GAN) | `4.0` | `1.0` | LPIPS is a natural-image *texture* prior. There is little true high-frequency content at low field, and a heavy perceptual weight makes the decoder invent texture. |
| `model.background_weight` (VQ-GAN) | — (plain L1) | `0.25` | The Rician noise floor dominates the pixel count. Unweighted L1 lets the codebook be spent on background. |
| `model.discriminator_iter_start` | `10000` | `20000` | 2D gives many more steps per epoch, and an early adversarial signal on noisy data hallucinates structure. |
| `model.n_codes` | `16384` | `8192` | Small cohorts do not fill a large codebook. Watch `train/perplexity`; raise only if it saturates near `n_codes`. |
| `model.n_hiddens` | `16` | `64` | 3D used 16 purely for memory. 2D can afford a wider encoder, and a tighter latent reconstruction matters more when SNR is already scarce. |
| `model.loss_type` (DDPM) | `l1` | `l1` (kept) | Robust to the heavy-tailed residuals a Rician floor produces — do **not** switch to `l2`. |
| `model.sampling_timesteps` | 500 (full) | `250` (DDIM) | A 2D model denoises every slice; per-volume inference is `n_slices × T` forward passes. |
| `model.min_snr_gamma` | — | `null`, try `5.0` | Min-SNR-γ weighting stabilises training on a few thousand slices. Off by default so the baseline matches the 3D behaviour. |
| `dataset.cond_noise_std` | — | `0.0`, try `0.02` | Rician noise augmentation on the *condition only*. Enable if test-time SNR may be below training SNR (e.g. training on 0.3 T M4Raw, testing on 64 mT). |
| `model.cond_drop_prob` / `cond_scale` | — | `0.0` / `1.0` | Set `0.1` / `1.5–2.0` to use classifier-free guidance, which sharpens output when conditioning is weak. |

Two further things worth knowing for low field, which are *not* handled automatically:

* **Bias field.** 64 mT images can carry strong intensity inhomogeneity. Percentile
  scaling does not remove it. If your data is not already bias-corrected, run N4
  (ANTs / SimpleITK) before this pipeline — otherwise the VQ-GAN codebook encodes
  scanner shading rather than anatomy.
* **Cross-field pairs.** If your "target" is a 3 T scan and your "condition" is a 64 mT
  scan (as in the paired 64 mT/3 T datasets), you are asking the model to do motion
  correction *and* field-strength translation *and* resolution super-resolution at
  once, and it will trade one against the other. Keep target and condition at the same
  field strength for the motion-correction task, and treat field translation as a
  separate experiment.

---

## Setup

```bash
pip install -r requirements.txt
```

On Windows, run everything from the repository root with:

```bat
set PYTHONPATH=%cd%
```

### Verify the install first

```bat
set PYTHONPATH=%cd% && python tools\smoke_test.py
```

No data, no checkpoints, no network access needed — it builds tiny versions of every
component and checks shapes and plumbing end to end (VQ-GAN both optimiser branches,
all three condition modes, both objectives, DDPM and DDIM sampling, classifier-free
guidance, the datasets, the volume→slices→volume round trip, and the motion simulator).
Runs in well under a minute on CPU. Exit code 0 means everything is wired correctly.

---

## Data format

Slices are indexed out of NIfTI volumes on the fly (nothing to pre-convert), and 2D
`.npy` / `.png` / `.tif` files are accepted too.

VQ-GAN training and DDPM pretraining — one folder of motion-free volumes:

```text
root_dir/
  sub_000001.nii.gz
  sub_000002.nii.gz
```

Conditional DDPM training — paired folders with matching file names:

```text
root_dir/
  target/
    sub_000001.nii.gz
  corrupted/
    sub_000001.nii.gz
```

Preprocessing: percentile-clip to `[-1, 1]` per image, centre crop/pad each slice to
`dataset.target_shape` (default `256×256`), drop slices below
`min_foreground_fraction`. **Target and condition are normalised independently** — at
inference you only have the corrupted image, so the model must never depend on the
clean image's scaling.

### Generating paired data (if you do not have it)

```bash
python tools/simulate_motion_2d.py --input D:\data\clean_volumes --output D:\data\cddpm_2d --pattern sudden --severity moderate --n-states 8 --pe-axis 0
```

`--pe-axis` must be the in-plane phase-encode axis of your slices (0 = rows, 1 =
columns) — get it wrong and the ghosting runs along the wrong direction. Patterns:
`sudden` (one abrupt move), `random_walk`, `periodic`. For M4Raw you can also build
real pairs from its repeated acquisitions instead of simulating.

---

## Training

### 1. Train the 2D VQ-GAN

```bat
set PYTHONPATH=%cd% && set PL_TORCH_DISTRIBUTED_BACKEND=gloo && python train\train_vqgan.py dataset=default dataset.root_dir="D:/data/VQGAN_training" model=vq_gan_2d model.gpus=1 model.default_root_dir="D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D" model.default_root_dir_postfix=lf_dataset model.precision=16 model.downsample=[4,4] model.embedding_dim=8 model.n_hiddens=64 model.num_groups=8 model.n_codes=8192 model.batch_size=16 model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=20000 model.perceptual_weight=1 model.image_gan_weight=1 model.gan_feat_weight=4 model.background_weight=0.25 model.accumulate_grad_batches=1 model.max_epochs=50
```

Note `model.lr` is auto-scaled by `accumulate × (gpus/8) × (batch/4)`, as in the 3D
repo — with `gpus=1, batch=16` the effective LR is `1.5e-4`.

Check it before moving on:

```bat
set PYTHONPATH=%cd% && python tools\test_vqgan_2d.py --ckpt "D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D/DEFAULT/lf_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt" --data "D:/data/VQGAN_training" --out "D:/out/vqgan_check"
```

Reconstruction SSIM should be ≳0.95 and perplexity should be a healthy fraction of
`n_codes`. If either is poor, fix stage 1 first — the DDPM can never beat its own
autoencoder.

### 2. Pretrain the 2D DDPM (unconditional, motion-free slices)

`diffusion_img_size = target_shape / downsample = 256/4 = 64`;
`diffusion_num_channels = embedding_dim = 8`.

```bat
set PYTHONPATH=%cd% && python train\train_ddpm.py model=ddpm dataset=default dataset.use_paired=false dataset.root_dir="D:/data/VQGAN_training" model.denoising_fn=Unet2D model.vqgan_ckpt="D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D/DEFAULT/lf_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt" model.diffusion_img_size=64 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=16 model.num_workers=4 model.gradient_accumulate_every=1 model.train_lr=1e-4 model.ema_decay=0.995 model.save_and_sample_every=1000 model.gpus=0 model.amp=True model.results_folder="D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d" model.results_folder_postfix=pretrain_lf
```

Resume from a milestone:

```bat
model.load_milestone="D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d/DEFAULT/pretrain_lf/model-120.pt"
```

### 3. Train the conditional 2D DDPM

```bat
set PYTHONPATH=%cd% && python train\train_ddpm.py model=ddpm dataset=default dataset.use_paired=true dataset.root_dir="D:/data/cddpm_2d" dataset.target_subdir=target dataset.condition_subdir=corrupted model.denoising_fn=Unet2D model.vqgan_ckpt="D:/Thesis/MOCO-CDDPM-2D/checkpoints/VQGAN_2D/DEFAULT/lf_dataset/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt" model.diffusion_img_size=64 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=16 model.num_workers=4 model.train_lr=1e-4 model.ema_decay=0.995 model.save_and_sample_every=1000 model.gpus=0 model.amp=True +model.fixed_sampling_seed=1 model.results_folder="D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d" model.results_folder_postfix=cddpm_lf model.load_milestone="D:/Thesis/MOCO-CDDPM-2D/checkpoints/ddpm_2d/DEFAULT/pretrain_lf/model-120.pt"
```

Stage 3 starts from the stage-2 weights. The condition-fusion layers do not exist in a
pretrained unconditional checkpoint, so they are initialised fresh — this is why
`Trainer.load` uses `strict=False`, exactly as the 3D repo did.

Each `save_and_sample_every` steps you get `<milestone>_comparison.png`
(condition ∥ sample ∥ target) and a row in `sampling_metrics.csv` with SSIM/PSNR for
both sample-vs-target and condition-vs-target — the second is your no-op baseline, and
the model is only earning its keep while the first is above it.

---

## Sampling

Whole volumes, slice by slice, written back as NIfTI:

```bat
set PYTHONPATH=%cd% && python tools\sample_2d.py model=ddpm dataset=default model.vqgan_ckpt="D:/.../latest_checkpoint.ckpt" model.diffusion_img_size=64 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.load_milestone="D:/.../model-300.pt" model.sampling_timesteps=250 +model.condition_dir="D:/data/test/corrupted" +model.target_dir="D:/data/test/target" +model.sample_batch_size=8 +model.output_dir="D:/out/inference"
```

Modes: `dataset.use_paired=true` (paired root), `+model.condition_dir=...` (folder,
`+model.target_dir` optional), `+model.fixed_condition_path=...` (single file).
Useful extras: `+model.use_ema=false`, `+model.max_files=2`, `model.cond_scale=1.5`,
`+model.fixed_sampling_seed=1`, `+model.save_nifti=false`.

## Evaluation

```bash
python evaluation/evaluate_2d.py --target D:\data\test\target --prediction D:\out\inference --condition D:\data\test\corrupted --suffix _sample --out D:\out\metrics_2d.csv
```

Reports SSIM, PSNR, MS-SSIM and LPIPS per volume (averaged over slices) plus the
uncorrected-input baseline.

---

## Shape cheat-sheet

```
image slice        [B, 1, 256, 256]     in [-1, 1]
VQ-GAN latent      [B, 8,  64,  64]     downsample [4, 4], embedding_dim 8
diffusion operates on the latent, normalised to [-1, 1] by codebook min/max
```

Changing `dataset.target_shape` or `model.downsample` means changing
`model.diffusion_img_size` to match — `GaussianDiffusion.forward` asserts this and
tells you the expected numbers.

## Acknowledgement

Built on [MoCo-cDDPM](https://github.com/alfattah129/MOCO-CDDPM), which is itself built
on **Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image
Synthesis** (https://arxiv.org/abs/2211.03364) and TATS.
