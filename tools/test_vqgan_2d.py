"""Sanity-check a trained 2D VQ-GAN before spending days on the DDPM.

Reports the reconstruction SSIM/PSNR and the codebook perplexity on a folder of
slices, and writes a few input/reconstruction PNGs.  If reconstruction SSIM is
not comfortably above what you expect from the diffusion stage, the latent
space is the bottleneck and there is no point training the DDPM yet.

Usage::

    set PYTHONPATH=%cd% && python tools\\test_vqgan_2d.py ^
        --ckpt "...\\latest_checkpoint.ckpt" ^
        --data "D:\\data\\val_volumes" --out "D:\\out\\vqgan_check" --n 64
"""

import argparse
import os

import numpy as np
import torch
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from dataset.default import Augment2D, DEFAULT2DDataset, Preprocess2D
from ddpm.diffusion import save_image_row
from vq_gan_2d.model import VQGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='vqgan_check')
    parser.add_argument('--n', type=int, default=64, help='slices to score')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--target-shape', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--slice-axis', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    model = VQGAN.load_from_checkpoint(args.ckpt).to(device).eval()
    print(f'latent: {model.embedding_dim} channels, {model.n_codes} codes')

    pre = Preprocess2D(target_shape=tuple(args.target_shape),
                       slice_axis=args.slice_axis)
    dataset = DEFAULT2DDataset(root_dir=args.data, pre=pre, augment=False,
                               aug=Augment2D())
    print(f'{len(dataset)} slices available')

    indices = np.linspace(0, len(dataset) - 1,
                          min(args.n, len(dataset))).astype(int)
    ssim_vals, psnr_vals, perplexities = [], [], []
    first_batch = None

    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start:start + args.batch_size]
            x = torch.stack([dataset[int(i)]['data'] for i in chunk]).to(device)

            z = model.pre_vq_conv(model.encoder(x))
            vq = model.codebook(z)
            x_recon = model.decoder(model.post_vq_conv(vq['embeddings']))

            a = ((x + 1) / 2).clamp(0, 1)
            b = ((x_recon + 1) / 2).clamp(0, 1)
            ssim_vals.append(structural_similarity_index_measure(
                b, a, data_range=1.0).item())
            psnr_vals.append(peak_signal_noise_ratio(
                b, a, data_range=1.0).item())
            perplexities.append(vq['perplexity'].item())

            if first_batch is None:
                first_batch = (x.cpu(), x_recon.cpu())

    print(f'reconstruction SSIM : {np.mean(ssim_vals):.4f}')
    print(f'reconstruction PSNR : {np.mean(psnr_vals):.2f} dB')
    print(f'codebook perplexity : {np.mean(perplexities):.1f} '
          f'(of {model.n_codes} codes)')

    if first_batch is not None:
        save_image_row(list(first_batch),
                       os.path.join(args.out, 'reconstruction.png'))
        print(f'wrote {os.path.join(args.out, "reconstruction.png")}')


if __name__ == '__main__':
    main()
