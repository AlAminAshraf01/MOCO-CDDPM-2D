"""Slice-wise SSIM / PSNR / MS-SSIM / LPIPS between two folders of volumes.

Replaces ``evaluation/ms-ssim_lpips_evaluation.ipynb`` with a script that runs
on 2D slices.  All metrics are computed per slice on [0, 1] data and averaged,
which is the convention the training/sampling CSVs use as well, so the numbers
are directly comparable.

Usage::

    python evaluation/evaluate_2d.py --target D:\\data\\test\\target \\
        --prediction D:\\out\\inference --suffix _sample \\
        --out D:\\out\\metrics.csv
"""

import argparse
import csv
import os

import numpy as np
import torch
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from dataset.default import Preprocess2D, load_volume_as_slices

NIFTI_EXTS = ('.nii', '.nii.gz')


def strip_ext(name):
    for ext in NIFTI_EXTS:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def index_folder(folder):
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(NIFTI_EXTS):
            out[strip_ext(name)] = os.path.join(folder, name)
    return out


def to01(tensor):
    return ((tensor.float() + 1) / 2).clamp(0., 1.)


def multiscale_ssim(pred, target):
    try:
        from torchmetrics.functional.image.ssim import \
            multiscale_structural_similarity_index_measure as ms_ssim
    except ImportError:
        return None
    try:
        return ms_ssim(pred, target, data_range=1.0).item()
    except (RuntimeError, ValueError):
        # MS-SSIM needs images larger than ~160 px for 5 scales
        return None


class LPIPSWrapper:
    """LPIPS on 1-channel slices by repeating the channel three times."""

    def __init__(self, device):
        self.model = None
        try:
            from torchmetrics.image.lpip import \
                LearnedPerceptualImagePatchSimilarity
            self.model = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg', normalize=True).to(device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover
            print(f'[warn] LPIPS unavailable ({exc}) -- skipping')
        self.device = device

    def __call__(self, pred, target, max_chunk=16):
        if self.model is None:
            return None
        values = []
        with torch.no_grad():
            for start in range(0, pred.shape[0], max_chunk):
                end = min(start + max_chunk, pred.shape[0])
                a = pred[start:end].repeat(1, 3, 1, 1).to(self.device)
                b = target[start:end].repeat(1, 3, 1, 1).to(self.device)
                values.append(self.model(a, b).item())
        return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--condition', default=None,
                        help='optional: also score the uncorrected input')
    parser.add_argument('--suffix', default='',
                        help='suffix the prediction files carry, e.g. _sample')
    parser.add_argument('--out', default='metrics_2d.csv')
    parser.add_argument('--target-shape', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--slice-axis', type=int, default=2)
    parser.add_argument('--percentiles', type=float, nargs=2,
                        default=[0.5, 99.5])
    parser.add_argument('--no-lpips', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre = Preprocess2D(target_shape=tuple(args.target_shape),
                       percentiles=tuple(args.percentiles),
                       slice_axis=args.slice_axis)

    targets = index_folder(args.target)
    predictions = index_folder(args.prediction)
    conditions = index_folder(args.condition) if args.condition else {}

    lpips = None if args.no_lpips else LPIPSWrapper(device)

    rows = []
    for key, target_path in targets.items():
        pred_key = key + args.suffix
        pred_path = predictions.get(pred_key) or predictions.get(key)
        if pred_path is None:
            print(f'[skip] no prediction for {key}')
            continue

        target_slices, _ = load_volume_as_slices(target_path, pre)
        pred_slices, _ = load_volume_as_slices(pred_path, pre)
        n = min(target_slices.shape[0], pred_slices.shape[0])
        t01, p01 = to01(target_slices[:n]), to01(pred_slices[:n])

        row = {
            'name': key,
            'n_slices': n,
            'ssim': structural_similarity_index_measure(
                p01, t01, data_range=1.0).item(),
            'psnr': peak_signal_noise_ratio(p01, t01, data_range=1.0).item(),
            'ms_ssim': multiscale_ssim(p01, t01),
            'lpips': lpips(p01, t01) if lpips else None,
        }

        cond_path = conditions.get(key)
        if cond_path:
            cond_slices, _ = load_volume_as_slices(cond_path, pre)
            c01 = to01(cond_slices[:n])
            row.update({
                'ssim_condition': structural_similarity_index_measure(
                    c01, t01, data_range=1.0).item(),
                'psnr_condition': peak_signal_noise_ratio(
                    c01, t01, data_range=1.0).item(),
            })

        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in row.items()})

    if not rows:
        raise SystemExit('nothing scored -- check the folders and --suffix')

    fieldnames = sorted({k for row in rows for k in row})
    fieldnames = ['name', 'n_slices'] + [f for f in fieldnames
                                         if f not in ('name', 'n_slices')]
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print('\n--- mean over volumes ---')
    for field in fieldnames:
        values = [r[field] for r in rows
                  if isinstance(r.get(field), (int, float))]
        if values and field != 'n_slices':
            print(f'{field}: {np.mean(values):.4f}')
    print(f'written to {args.out}')


if __name__ == '__main__':
    main()
