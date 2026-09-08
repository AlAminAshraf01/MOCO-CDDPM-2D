"""End-to-end shape/plumbing check. No data, no checkpoints, no network.

Run this first in your torch environment -- it exercises every path the three
training stages use, on tiny tensors, in well under a minute on CPU::

    set PYTHONPATH=%cd% && python tools\\smoke_test.py

It checks:
  1. the 2D VQ-GAN forward / encode / decode and both optimiser branches
  2. Unet2D in all three condition modes, with and without slice positions
  3. GaussianDiffusion loss, DDPM sampling and DDIM sampling in latent space
  4. the full VQ-GAN + diffusion latent round trip (image in, image out)
  5. classifier-free guidance
  6. the slice dataset, the paired dataset, and the volume -> slices -> volume
     round trip, on synthetic NIfTI files
  7. the k-space motion simulator
"""

import os
import shutil
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from ddpm import GaussianDiffusion, Unet2D
from ddpm.diffusion import save_image_row
from vq_gan_2d.model import VQGAN

PASS, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        PASS.append(name)
        print(f'  PASS  {name}' + (f'   [{detail}]' if detail else ''))
    except Exception as exc:  # noqa: BLE001 - a smoke test reports everything
        FAIL.append((name, exc))
        print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')


def tiny_vqgan_cfg(image_size=64, downsample=(4, 4), n_hiddens=8,
                   embedding_dim=4, n_codes=64):
    return OmegaConf.create({
        'dataset': {'name': 'DEFAULT', 'image_channels': 1},
        'model': {
            'seed': 0, 'batch_size': 2, 'num_workers': 0, 'gpus': 0,
            'accumulate_grad_batches': 1, 'lr': 3e-4,
            'downsample': list(downsample), 'embedding_dim': embedding_dim,
            'n_codes': n_codes, 'n_hiddens': n_hiddens, 'num_groups': 4,
            'norm_type': 'group', 'padding_type': 'replicate',
            'restart_thres': 1.0, 'no_random_restart': False,
            'disc_channels': 16, 'disc_layers': 3, 'disc_loss_type': 'hinge',
            'discriminator_iter_start': 0, 'image_gan_weight': 1.0,
            'gan_feat_weight': 1.0, 'l1_weight': 4.0,
            'perceptual_weight': 0.0,          # keeps LPIPS (and its download) out
            'background_weight': 0.25, 'foreground_threshold': -0.8,
            'adam_capturable': False,
            'default_root_dir': '.', 'default_root_dir_postfix': '',
        },
    })


def main():
    torch.manual_seed(0)
    device = torch.device('cpu')
    image_size, latent_size, latent_ch = 64, 16, 4

    print('\n1. VQ-GAN 2D')
    cfg = tiny_vqgan_cfg(image_size=image_size, downsample=(4, 4),
                         embedding_dim=latent_ch)
    vqgan = VQGAN(cfg).to(device).eval()
    x = torch.randn(2, 1, image_size, image_size).clamp(-1, 1)

    def _vq_forward():
        _, x_recon, vq_out, _ = vqgan(x)
        assert x_recon.shape == x.shape, x_recon.shape
        return f'recon {tuple(x_recon.shape)}, perplexity {vq_out["perplexity"]:.1f}'
    check('vqgan forward', _vq_forward)

    def _vq_latent():
        h = vqgan.encode(x, quantize=False, include_embeddings=True)
        assert h.shape == (2, latent_ch, latent_size, latent_size), h.shape
        back = vqgan.decode(h, quantize=True)
        assert back.shape == x.shape, back.shape
        return f'latent {tuple(h.shape)}'
    check('vqgan encode/decode', _vq_latent)

    def _vq_opt():
        vqgan.train()
        out0 = vqgan(x, optimizer_idx=0)
        assert len(out0) == 6
        loss_d = vqgan(x, optimizer_idx=1)
        assert loss_d.ndim == 0
        vqgan.eval()
        return f'g-loss terms {len(out0)}, d-loss {loss_d.item():.4f}'
    check('vqgan generator + discriminator branches', _vq_opt)

    def _vq_optimizers():
        opts, _ = vqgan.configure_optimizers()
        assert len(opts) == 2
        return '2 optimizers'
    check('vqgan configure_optimizers', _vq_optimizers)

    print('\n2. Unet2D')
    latent = torch.randn(2, latent_ch, latent_size, latent_size)
    t = torch.randint(0, 500, (2,)).float()
    cond = torch.randn_like(latent)
    pos = torch.rand(2)

    for mode in ('gated', 'concat', 'both'):
        def _unet(mode=mode):
            net = Unet2D(dim=32, dim_mults=(1, 2, 4), channels=latent_ch,
                         cond_mode=mode, resnet_groups=8)
            out = net(latent, t, cond=cond, slice_pos=pos)
            assert out.shape == latent.shape, out.shape
            n_params = sum(p.numel() for p in net.parameters())
            return f'{tuple(out.shape)}, {n_params / 1e6:.2f}M params'
        check(f'unet2d cond_mode={mode}', _unet)

    def _unet_uncond():
        net = Unet2D(dim=32, dim_mults=(1, 2, 4), channels=latent_ch,
                     cond_mode='gated')
        out = net(latent, t)              # no cond, no slice_pos
        assert out.shape == latent.shape
        return 'unconditional path OK'
    check('unet2d without condition (pretraining path)', _unet_uncond)

    def _unet_cfg():
        net = Unet2D(dim=32, dim_mults=(1, 2, 4), channels=latent_ch)
        guided = net.forward_with_cond_scale(latent, t, cond=cond,
                                             slice_pos=pos, cond_scale=2.0)
        plain = net.forward_with_cond_scale(latent, t, cond=cond,
                                            slice_pos=pos, cond_scale=1.0)
        assert guided.shape == latent.shape
        assert not torch.allclose(guided, plain), 'cond_scale had no effect'
        return 'guidance changes the output'
    check('classifier-free guidance', _unet_cfg)

    print('\n3. GaussianDiffusion (pure latent, no VQ-GAN)')
    net = Unet2D(dim=32, dim_mults=(1, 2, 4), channels=latent_ch)
    diffusion = GaussianDiffusion(net, image_size=latent_size,
                                  channels=latent_ch, timesteps=50,
                                  sampling_timesteps=10, loss_type='l1',
                                  cond_drop_prob=0.1).to(device)

    def _loss():
        loss = diffusion(latent, cond=cond, slice_pos=pos)
        loss.backward()
        grads = sum(1 for p in net.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads > 0, 'no gradients reached the network'
        return f'loss {loss.item():.4f}, {grads} tensors got gradient'
    check('diffusion loss + backward', _loss)

    def _objectives():
        out = {}
        for objective in ('pred_noise', 'pred_x0'):
            d = GaussianDiffusion(Unet2D(dim=32, dim_mults=(1, 2),
                                         channels=latent_ch),
                                  image_size=latent_size, channels=latent_ch,
                                  timesteps=50, objective=objective)
            out[objective] = round(d(latent, cond=cond).item(), 4)
        return str(out)
    check('objectives pred_noise / pred_x0', _objectives)

    def _min_snr():
        d = GaussianDiffusion(Unet2D(dim=32, dim_mults=(1, 2),
                                     channels=latent_ch),
                              image_size=latent_size, channels=latent_ch,
                              timesteps=50, min_snr_gamma=5.0)
        assert d.loss_weight.min() < 1.0, 'min-snr weighting is flat'
        return f'weights in [{d.loss_weight.min():.3f}, {d.loss_weight.max():.3f}]'
    check('min-SNR-gamma weighting', _min_snr)

    def _ddim():
        out = diffusion.sample(cond=cond, slice_pos=pos, use_ddim=True,
                               progress=False)
        assert out.shape == (2, latent_ch, latent_size, latent_size), out.shape
        return f'{tuple(out.shape)} in {diffusion.sampling_timesteps} steps'
    check('DDIM sampling', _ddim)

    def _ddpm():
        out = diffusion.sample(cond=cond, slice_pos=pos, use_ddim=False,
                               progress=False)
        assert out.shape == (2, latent_ch, latent_size, latent_size)
        return f'{diffusion.num_timesteps} steps'
    check('ancestral (DDPM) sampling', _ddpm)

    print('\n4. Full latent pipeline (image -> latent -> image)')

    def _full():
        d = GaussianDiffusion(Unet2D(dim=32, dim_mults=(1, 2),
                                     channels=latent_ch),
                              image_size=latent_size, channels=latent_ch,
                              timesteps=50, sampling_timesteps=5)
        # attach the VQ-GAN built above instead of loading a checkpoint
        d.vqgan = vqgan
        d._vq_embedding_min = vqgan.codebook.embeddings.min().clone()
        d._vq_embedding_max = vqgan.codebook.embeddings.max().clone()
        d.train()                                  # must NOT wake the VQ-GAN
        assert not d.vqgan.training, 'VQ-GAN was put back into train mode'

        cond_image = torch.randn(2, 1, image_size, image_size).clamp(-1, 1)
        cond_latent = d.encode_condition(cond_image)
        assert cond_latent.shape == (2, latent_ch, latent_size, latent_size)
        # an already-encoded latent must pass through untouched
        assert d.encode_condition(cond_latent) is cond_latent

        loss = d(cond_image, cond=cond_latent, slice_pos=torch.rand(2))
        out = d.sample(cond=cond_latent, slice_pos=torch.rand(2), progress=False)
        assert out.shape == cond_image.shape, out.shape
        return f'image {tuple(cond_image.shape)} -> {tuple(out.shape)}, loss {loss.item():.4f}'
    check('vqgan + diffusion round trip', _full)

    def _shape_guard():
        d = GaussianDiffusion(Unet2D(dim=32, dim_mults=(1, 2),
                                     channels=latent_ch),
                              image_size=8, channels=latent_ch, timesteps=50)
        try:
            d(latent)                      # latent is 16x16, config says 8x8
        except AssertionError as exc:
            assert 'diffusion_num_channels' in str(exc)
            return 'mismatch is reported clearly'
        raise RuntimeError('a latent/config mismatch went undetected')
    check('latent/config mismatch guard', _shape_guard)

    print('\n5. Datasets and NIfTI round trip')
    try:
        import nibabel as nib
    except ImportError:
        print('  SKIP  dataset checks (nibabel not installed)')
        nib = None

    tmp = tempfile.mkdtemp(prefix='moco2d_smoke_')
    if nib is not None:
        from dataset.default import (Augment2D, DEFAULT2DDataset,
                                     DEFAULT2DPairedDataset, Preprocess2D,
                                     load_volume_as_slices,
                                     save_slices_as_volume)

        rng = np.random.default_rng(0)
        plain = os.path.join(tmp, 'plain')
        paired_t = os.path.join(tmp, 'paired', 'target')
        paired_c = os.path.join(tmp, 'paired', 'corrupted')
        for folder in (plain, paired_t, paired_c):
            os.makedirs(folder, exist_ok=True)

        def make_volume(path, shape=(72, 68, 6)):
            vol = rng.gamma(2.0, 20.0, size=shape).astype(np.float32)
            vol[10:60, 10:55, 1:5] += 400.0        # a "brain"
            nib.save(nib.Nifti1Image(vol, np.eye(4)), path)

        for i in range(3):
            make_volume(os.path.join(plain, f'sub_{i}.nii.gz'))
            make_volume(os.path.join(paired_t, f'sub_{i}.nii.gz'))
            make_volume(os.path.join(paired_c, f'sub_{i}.nii.gz'))

        pre = Preprocess2D(target_shape=(64, 64), min_foreground_fraction=0.0)
        aug = Augment2D()

        def _ds():
            ds = DEFAULT2DDataset(plain, pre=pre, augment=False, aug=aug,
                                  verbose=False)
            item = ds[0]
            assert item['data'].shape == (1, 64, 64), item['data'].shape
            assert -1.001 <= item['data'].min() <= item['data'].max() <= 1.001
            assert 0.0 <= float(item['slice_pos']) <= 1.0
            return f'{len(ds)} slices from 3 volumes, sample {tuple(item["data"].shape)}'
        check('slice dataset', _ds)

        def _ds_aug():
            ds = DEFAULT2DDataset(plain, pre=pre, augment=True, aug=aug,
                                  verbose=False)
            assert ds[0]['data'].shape == (1, 64, 64)
            return 'augmented sample OK'
        check('slice dataset with augmentation', _ds_aug)

        def _paired():
            ds = DEFAULT2DPairedDataset(os.path.join(tmp, 'paired'), pre=pre,
                                        augment=True, aug=aug, verbose=False)
            item = ds[0]
            assert item['data'].shape == item['cond'].shape == (1, 64, 64)
            return f'{len(ds)} paired slices'
        check('paired dataset', _paired)

        def _filter():
            strict = Preprocess2D(target_shape=(64, 64),
                                  min_foreground_fraction=0.5)
            ds = DEFAULT2DDataset(plain, pre=strict, augment=False, aug=aug,
                                  verbose=False)
            loose = DEFAULT2DDataset(plain, pre=pre, augment=False, aug=aug,
                                     verbose=False)
            assert len(ds) < len(loose), 'foreground filter did nothing'
            return f'{len(loose)} -> {len(ds)} slices'
        check('empty-slice rejection', _filter)

        def _roundtrip():
            path = os.path.join(plain, 'sub_0.nii.gz')
            slices, meta = load_volume_as_slices(path, pre)
            assert slices.shape[1:] == (1, 64, 64), slices.shape
            out = os.path.join(tmp, 'restored.nii.gz')
            save_slices_as_volume(slices, meta, out)
            restored = np.asanyarray(nib.load(out).dataobj)
            original = np.asanyarray(nib.load(path).dataobj)
            assert restored.shape == original.shape, (restored.shape,
                                                      original.shape)
            return f'{slices.shape[0]} slices -> {restored.shape}'
        check('volume -> slices -> volume', _roundtrip)

        def _png():
            slices, _ = load_volume_as_slices(
                os.path.join(plain, 'sub_0.nii.gz'), pre)
            out = os.path.join(tmp, 'grid.png')
            save_image_row([slices[:3], slices[:3]], out, max_items=3)
            assert os.path.exists(out)
            return 'comparison PNG written'
        check('PNG comparison grid', _png)

    print('\n6. Motion simulation')

    def _sim():
        from tools.simulate_motion_2d import corrupt_slice
        img = np.zeros((64, 64))
        img[16:48, 20:44] = 1.0
        out = corrupt_slice(img, n_states=8, pattern='sudden',
                            severity='moderate', pe_axis=0,
                            rng=np.random.default_rng(0))
        assert out.shape == img.shape
        residual = float(np.abs(out - img).mean())
        assert residual > 1e-3, 'no artefact produced'
        return f'mean |corrupted - clean| = {residual:.4f}'
    check('k-space motion corruption', _sim)

    shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 60)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    for name, exc in FAIL:
        print(f'  - {name}: {exc}')
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
