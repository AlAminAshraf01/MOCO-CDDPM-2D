"""2D slice datasets for MoCo-cDDPM-2D.

The 3D repository fed whole volumes through torchio:

    tio.RescaleIntensity(out_min_max=(-1, 1))   # global min/max
    tio.CropOrPad((160, 160, 64))               # pads with 0

Both of those are poor choices for ultra-low-field data, so the 2D pipeline
changes them deliberately:

1. **Percentile intensity scaling instead of min/max.**  At 64 mT / 0.3 T a
   single hot voxel (flow, fat, an RF spike) sets the max and squashes the
   brain into a fraction of the [-1, 1] range.  We clip at
   ``percentiles`` (default 0.5/99.5) before scaling.
2. **Padding with -1, not 0.**  After rescaling to [-1, 1], zero is mid-grey.
   Padding air with mid-grey taught the 3D model to draw bright borders; -1 is
   background.
3. **Slice-level indexing.**  One training sample is one slice, which is the
   whole point of the 2D port: a low-field cohort of 100 volumes x 18 slices
   becomes 1800 samples instead of 100.
4. **Empty-slice rejection.**  End slices of a low-field stack are pure Rician
   noise; ``min_foreground_fraction`` drops them.
5. **Normalisation statistics are per-image, never shared between the target
   and the condition** -- at inference you only have the corrupted image, so
   the model must never depend on the clean image's scaling.

Both the target and the condition of a pair receive *identical* geometric
augmentation.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # optional -- only needed for NIfTI input
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

NIFTI_EXTS = ('.nii', '.nii.gz')
ARRAY_EXTS = ('.npy',)
IMAGE_EXTS = ('.png', '.tif', '.tiff')
ALL_EXTS = NIFTI_EXTS + ARRAY_EXTS + IMAGE_EXTS


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Preprocess2D:
    """Everything that has to match between training and inference."""

    target_shape: Tuple[int, int] = (256, 256)
    percentiles: Tuple[float, float] = (0.5, 99.5)
    norm_scope: str = 'volume'          # 'volume' | 'slice'
    pad_value: float = -1.0
    slice_axis: int = 2
    min_foreground_fraction: float = 0.0
    slice_range: Tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_cfg(cls, dataset_cfg) -> 'Preprocess2D':
        def g(name, default):
            value = getattr(dataset_cfg, name, default)
            return default if value is None else value

        shape = tuple(int(v) for v in g('target_shape', (256, 256)))
        pct = tuple(float(v) for v in g('percentiles', (0.5, 99.5)))
        srange = tuple(float(v) for v in g('slice_range', (0.0, 1.0)))
        return cls(
            target_shape=shape,
            percentiles=pct,
            norm_scope=str(g('norm_scope', 'volume')),
            pad_value=float(g('pad_value', -1.0)),
            slice_axis=int(g('slice_axis', 2)),
            min_foreground_fraction=float(g('min_foreground_fraction', 0.0)),
            slice_range=srange,
        )


@dataclass
class Augment2D:
    """Paired-safe 2D augmentation.

    ``flip_lr`` is the only flip that is anatomically defensible for brain
    slices.  Rotation/translation/scale are small on purpose: low-field images
    are already blurry and interpolation costs resolution you cannot spare.
    ``cond_noise_std`` adds *Rician* noise to the condition only -- switch it on
    when test-time SNR may be lower than training SNR.
    """

    flip_lr_prob: float = 0.5
    max_rotation_deg: float = 10.0
    max_translation_frac: float = 0.03
    scale_range: Tuple[float, float] = (0.97, 1.03)
    cond_noise_std: float = 0.0
    cond_noise_prob: float = 0.0
    seed: Optional[int] = None

    @classmethod
    def from_cfg(cls, dataset_cfg) -> 'Augment2D':
        def g(name, default):
            value = getattr(dataset_cfg, name, default)
            return default if value is None else value

        return cls(
            flip_lr_prob=float(g('flip_lr_prob', 0.5)),
            max_rotation_deg=float(g('max_rotation_deg', 10.0)),
            max_translation_frac=float(g('max_translation_frac', 0.03)),
            scale_range=tuple(float(v) for v in g('scale_range', (0.97, 1.03))),
            cond_noise_std=float(g('cond_noise_std', 0.0)),
            cond_noise_prob=float(g('cond_noise_prob', 0.0)),
        )


# --------------------------------------------------------------------------- #
# intensity + geometry helpers
# --------------------------------------------------------------------------- #
def percentile_bounds(array: np.ndarray,
                      percentiles: Tuple[float, float]) -> Tuple[float, float]:
    lo_p, hi_p = percentiles
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, lo_p))
    hi = float(np.percentile(finite, hi_p))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def rescale_to_pm1(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=lo,
                          posinf=hi, neginf=lo)
    array = np.clip(array, lo, hi)
    return (array - lo) / (hi - lo) * 2.0 - 1.0


def crop_or_pad_2d(array: np.ndarray, target_shape: Tuple[int, int],
                   pad_value: float = -1.0) -> np.ndarray:
    """Centre crop / centre pad a 2D array to ``target_shape``."""
    out = array
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current == target:
            continue
        if current > target:
            start = (current - target) // 2
            out = np.take(out, np.arange(start, start + target), axis=axis)
        else:
            total = target - current
            before = total // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, total - before)
            out = np.pad(out, pad, mode='constant', constant_values=pad_value)
    return out


def _affine_2d(array: np.ndarray, angle_deg: float, scale: float,
               shift: Tuple[float, float], cval: float) -> np.ndarray:
    from scipy import ndimage

    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    matrix = np.array([[cos, -sin], [sin, cos]], dtype=np.float64) / scale
    centre = np.array(array.shape, dtype=np.float64) / 2.0
    offset = centre - matrix @ centre + np.asarray(shift, dtype=np.float64)
    return ndimage.affine_transform(array, matrix, offset=offset, order=1,
                                    mode='constant', cval=cval,
                                    output=np.float32)


def apply_augmentation(images: Sequence[np.ndarray], aug: Augment2D,
                       pad_value: float, rng: random.Random,
                       cond_index: Optional[int] = None) -> List[np.ndarray]:
    """Apply the *same* geometry to every image in ``images``."""
    out = [np.asarray(im, dtype=np.float32) for im in images]

    if aug.flip_lr_prob > 0 and rng.random() < aug.flip_lr_prob:
        out = [np.ascontiguousarray(im[:, ::-1]) for im in out]

    needs_affine = (aug.max_rotation_deg > 0 or aug.max_translation_frac > 0
                    or aug.scale_range != (1.0, 1.0))
    if needs_affine:
        angle = rng.uniform(-aug.max_rotation_deg, aug.max_rotation_deg)
        scale = rng.uniform(*aug.scale_range)
        max_shift = aug.max_translation_frac * max(out[0].shape)
        shift = (rng.uniform(-max_shift, max_shift),
                 rng.uniform(-max_shift, max_shift))
        if abs(angle) > 1e-3 or abs(scale - 1.0) > 1e-3 or max(map(abs, shift)) > 1e-3:
            out = [_affine_2d(im, angle, scale, shift, pad_value) for im in out]

    if (cond_index is not None and aug.cond_noise_std > 0
            and rng.random() < aug.cond_noise_prob):
        sigma = rng.uniform(0.0, aug.cond_noise_std)
        img01 = (out[cond_index] + 1.0) / 2.0
        real = img01 + np.random.normal(0.0, sigma, img01.shape)
        imag = np.random.normal(0.0, sigma, img01.shape)
        img01 = np.sqrt(real ** 2 + imag ** 2)          # Rician magnitude
        out[cond_index] = np.clip(img01 * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    return out


# --------------------------------------------------------------------------- #
# file readers
# --------------------------------------------------------------------------- #
def _is_nifti(path: str) -> bool:
    return path.lower().endswith(NIFTI_EXTS)


def _read_2d_file(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(ARRAY_EXTS):
        array = np.load(path)
    elif lower.endswith(IMAGE_EXTS):
        import imageio.v2 as imageio
        array = np.asarray(imageio.imread(path))
    else:
        raise ValueError(f'Not a supported 2D file: {path}')
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D array in {path}, got {array.shape}')
    return array


def _nifti_shape(path: str) -> Tuple[int, ...]:
    if nib is None:
        raise ImportError('nibabel is required to read NIfTI files')
    return tuple(int(s) for s in nib.load(path).shape[:3])


def _nifti_volume(path: str) -> np.ndarray:
    if nib is None:
        raise ImportError('nibabel is required to read NIfTI files')
    vol = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    return np.squeeze(vol)


def _nifti_slice(path: str, index: int, slice_axis: int) -> np.ndarray:
    """Read a single slice; ``.nii`` is memory-mapped, ``.nii.gz`` is not."""
    if nib is None:
        raise ImportError('nibabel is required to read NIfTI files')
    dataobj = nib.load(path).dataobj
    selector: List[object] = [slice(None)] * 3
    selector[slice_axis] = index
    array = np.asarray(dataobj[tuple(selector)]).astype(np.float32)
    return np.squeeze(array)


# --------------------------------------------------------------------------- #
# volume statistics cache
# --------------------------------------------------------------------------- #
def _cache_path(root_dir: str, tag: str) -> str:
    digest = hashlib.md5(os.path.abspath(root_dir).encode('utf-8')).hexdigest()[:10]
    name = f'.moco2d_index_{tag}_{digest}.json'
    try:
        os.makedirs(root_dir, exist_ok=True)
        candidate = os.path.join(root_dir, name)
        with open(candidate, 'a'):
            pass
        return candidate
    except OSError:
        return os.path.join(tempfile.gettempdir(), name)


def _file_key(path: str) -> str:
    stat = os.stat(path)
    return f'{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime)}'


def scan_volume(path: str, pre: Preprocess2D) -> Dict[str, object]:
    """One pass over a volume: percentile bounds + per-slice foreground."""
    volume = _nifti_volume(path)
    if volume.ndim == 2:
        volume = volume[..., None]
    lo, hi = percentile_bounds(volume, pre.percentiles)
    n_slices = volume.shape[pre.slice_axis]
    threshold = lo + 0.10 * (hi - lo)
    fractions = []
    for k in range(n_slices):
        sl = np.take(volume, k, axis=pre.slice_axis)
        fractions.append(float(np.mean(sl > threshold)))
    return {'lo': lo, 'hi': hi, 'n_slices': int(n_slices),
            'foreground': fractions}


def build_volume_stats(paths: Sequence[str], pre: Preprocess2D,
                       cache_file: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, Dict[str, object]]:
    cache: Dict[str, Dict[str, object]] = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as handle:
                cache = json.load(handle)
        except (OSError, ValueError):
            cache = {}

    stats: Dict[str, Dict[str, object]] = {}
    missing = [p for p in paths if _file_key(p) not in cache]
    iterator = missing
    if verbose and missing:
        try:
            from tqdm import tqdm
            iterator = tqdm(missing, desc='indexing volumes')
        except ImportError:
            pass
    for path in iterator:
        cache[_file_key(path)] = scan_volume(path, pre)

    for path in paths:
        stats[path] = cache[_file_key(path)]

    if cache_file and missing:
        try:
            with open(cache_file, 'w') as handle:
                json.dump(cache, handle)
        except OSError:
            pass
    return stats


# --------------------------------------------------------------------------- #
# slice index
# --------------------------------------------------------------------------- #
@dataclass
class SliceRecord:
    path: str
    cond_path: Optional[str]
    slice_index: int
    n_slices: int
    lo: float
    hi: float
    cond_lo: float = 0.0
    cond_hi: float = 1.0
    is_volume: bool = True

    @property
    def slice_pos(self) -> float:
        if self.n_slices <= 1:
            return 0.5
        return self.slice_index / (self.n_slices - 1)


def _keep_slice(index: int, n_slices: int, foreground: Sequence[float],
                pre: Preprocess2D) -> bool:
    lo_frac, hi_frac = pre.slice_range
    if n_slices > 1:
        pos = index / (n_slices - 1)
        if pos < lo_frac or pos > hi_frac:
            return False
    if pre.min_foreground_fraction > 0 and index < len(foreground):
        if foreground[index] < pre.min_foreground_fraction:
            return False
    return True


def list_files(directory: str, exts: Sequence[str] = ALL_EXTS) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
    names = [n for n in os.listdir(directory) if n.lower().endswith(tuple(exts))]
    return sorted(os.path.join(directory, n) for n in names)


def split_sequence_deterministically(items: Sequence, val_fraction: float = 0.1,
                                     seed: int = 42) -> Tuple[List, List]:
    """Unchanged from the 3D repo, but always called on *file* lists.

    Splitting on files rather than slices is what keeps slices of the same
    subject out of both the train and the validation set.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError(f'val_fraction must be in [0, 1), got {val_fraction}')

    items = list(items)
    if len(items) <= 1 or val_fraction == 0:
        return items, []

    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)

    val_count = int(round(len(items) * val_fraction))
    val_count = max(1, min(len(items) - 1, val_count))

    val_indices = set(indices[:val_count])
    train_items = [items[i] for i in range(len(items)) if i not in val_indices]
    val_items = [items[i] for i in range(len(items)) if i in val_indices]
    return train_items, val_items


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class _Base2DDataset(Dataset):
    def __init__(self, pre: Preprocess2D, aug: Optional[Augment2D]):
        super().__init__()
        self.pre = pre
        self.aug = aug
        self.records: List[SliceRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: str, index: int, is_volume: bool,
              lo: float, hi: float) -> np.ndarray:
        if is_volume:
            array = _nifti_slice(path, index, self.pre.slice_axis)
        else:
            array = _read_2d_file(path)
        if self.pre.norm_scope == 'slice':
            lo, hi = percentile_bounds(array, self.pre.percentiles)
        array = rescale_to_pm1(array, lo, hi)
        return crop_or_pad_2d(array, self.pre.target_shape, self.pre.pad_value)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array,
                                                     dtype=np.float32))[None]


class DEFAULT2DDataset(_Base2DDataset):
    """Unpaired slices -- for VQ-GAN training and DDPM pretraining."""

    def __init__(self, root_dir: str, pre: Optional[Preprocess2D] = None,
                 augment: bool = True, aug: Optional[Augment2D] = None,
                 file_paths: Optional[Sequence[str]] = None,
                 cache_dir: Optional[str] = None, verbose: bool = True):
        pre = pre or Preprocess2D()
        super().__init__(pre, aug if augment else None)
        self.root_dir = root_dir
        self.file_paths = (list(file_paths) if file_paths is not None
                           else list_files(root_dir))
        if not self.file_paths:
            raise RuntimeError(f'No supported files found in {root_dir}')
        self.records = self._build_index(cache_dir or root_dir, verbose)
        if not self.records:
            raise RuntimeError(
                f'No slices survived filtering in {root_dir}; relax '
                f'min_foreground_fraction / slice_range')

    def _build_index(self, cache_dir: str, verbose: bool) -> List[SliceRecord]:
        volumes = [p for p in self.file_paths if _is_nifti(p)]
        planar = [p for p in self.file_paths if not _is_nifti(p)]

        records: List[SliceRecord] = []
        if volumes:
            stats = build_volume_stats(
                volumes, self.pre, _cache_path(cache_dir, 'single'), verbose)
            for path in volumes:
                info = stats[path]
                n_slices = int(info['n_slices'])
                foreground = info['foreground']
                for k in range(n_slices):
                    if not _keep_slice(k, n_slices, foreground, self.pre):
                        continue
                    records.append(SliceRecord(
                        path=path, cond_path=None, slice_index=k,
                        n_slices=n_slices, lo=float(info['lo']),
                        hi=float(info['hi']), is_volume=True))
        for path in planar:
            records.append(SliceRecord(path=path, cond_path=None, slice_index=0,
                                       n_slices=1, lo=0.0, hi=1.0,
                                       is_volume=False))
        return records

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        image = self._load(rec.path, rec.slice_index, rec.is_volume,
                           rec.lo, rec.hi)
        if self.aug is not None:
            rng = random.Random(torch.randint(0, 2 ** 31 - 1, (1,)).item())
            image = apply_augmentation([image], self.aug,
                                       self.pre.pad_value, rng)[0]
        return {
            'data': self._to_tensor(image),
            'slice_pos': torch.tensor(rec.slice_pos, dtype=torch.float32),
            'path': rec.path,
            'slice_index': rec.slice_index,
        }


class DEFAULT2DPairedDataset(_Base2DDataset):
    """Paired (motion-free target, motion-corrupted condition) slices."""

    def __init__(self, root_dir: str, target_subdir: str = 'target',
                 condition_subdir: str = 'corrupted',
                 pre: Optional[Preprocess2D] = None, augment: bool = True,
                 aug: Optional[Augment2D] = None,
                 file_names: Optional[Sequence[str]] = None,
                 cache_dir: Optional[str] = None, verbose: bool = True):
        pre = pre or Preprocess2D()
        super().__init__(pre, aug if augment else None)
        self.root_dir = root_dir
        self.target_dir = os.path.join(root_dir, target_subdir)
        self.condition_dir = os.path.join(root_dir, condition_subdir)

        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(
                f'Target directory not found: {self.target_dir}')
        if not os.path.isdir(self.condition_dir):
            raise FileNotFoundError(
                f'Condition directory not found: {self.condition_dir}')

        self.file_names = (list(file_names) if file_names is not None
                           else self.get_common_file_names())
        self.records = self._build_index(cache_dir or root_dir, verbose)
        if not self.records:
            raise RuntimeError(
                f'No paired slices survived filtering in {root_dir}')

    def get_common_file_names(self) -> List[str]:
        def names(directory):
            return {n for n in os.listdir(directory)
                    if n.lower().endswith(ALL_EXTS)}

        common = sorted(names(self.target_dir) & names(self.condition_dir))
        if not common:
            raise RuntimeError(f'No matching files between {self.target_dir} '
                               f'and {self.condition_dir}')
        return common

    def _build_index(self, cache_dir: str, verbose: bool) -> List[SliceRecord]:
        target_paths = [os.path.join(self.target_dir, n) for n in self.file_names]
        cond_paths = [os.path.join(self.condition_dir, n) for n in self.file_names]

        vol_targets = [p for p in target_paths if _is_nifti(p)]
        vol_conds = [p for p in cond_paths if _is_nifti(p)]
        tgt_stats = (build_volume_stats(vol_targets, self.pre,
                                        _cache_path(cache_dir, 'tgt'), verbose)
                     if vol_targets else {})
        cond_stats = (build_volume_stats(vol_conds, self.pre,
                                         _cache_path(cache_dir, 'cond'), verbose)
                      if vol_conds else {})

        records: List[SliceRecord] = []
        for tgt, cond in zip(target_paths, cond_paths):
            if _is_nifti(tgt):
                t_info, c_info = tgt_stats[tgt], cond_stats[cond]
                n_slices = min(int(t_info['n_slices']), int(c_info['n_slices']))
                foreground = t_info['foreground']
                for k in range(n_slices):
                    if not _keep_slice(k, n_slices, foreground, self.pre):
                        continue
                    records.append(SliceRecord(
                        path=tgt, cond_path=cond, slice_index=k,
                        n_slices=n_slices,
                        lo=float(t_info['lo']), hi=float(t_info['hi']),
                        cond_lo=float(c_info['lo']), cond_hi=float(c_info['hi']),
                        is_volume=True))
            else:
                records.append(SliceRecord(
                    path=tgt, cond_path=cond, slice_index=0, n_slices=1,
                    lo=0.0, hi=1.0, cond_lo=0.0, cond_hi=1.0, is_volume=False))
        return records

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        target = self._load(rec.path, rec.slice_index, rec.is_volume,
                            rec.lo, rec.hi)
        condition = self._load(rec.cond_path, rec.slice_index, rec.is_volume,
                               rec.cond_lo, rec.cond_hi)
        if self.aug is not None:
            rng = random.Random(torch.randint(0, 2 ** 31 - 1, (1,)).item())
            target, condition = apply_augmentation(
                [target, condition], self.aug, self.pre.pad_value, rng,
                cond_index=1)
        return {
            'data': self._to_tensor(target),
            'cond': self._to_tensor(condition),
            'slice_pos': torch.tensor(rec.slice_pos, dtype=torch.float32),
            'path': rec.path,
            'cond_path': rec.cond_path,
            'slice_index': rec.slice_index,
        }


# --------------------------------------------------------------------------- #
# inference helper -- reuses the exact training preprocessing
# --------------------------------------------------------------------------- #
def load_volume_as_slices(path: str, pre: Preprocess2D,
                          keep_all_slices: bool = True
                          ) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return ``([N, 1, H, W]`` in [-1, 1], metadata) for a NIfTI volume.

    The metadata carries everything ``save_slices_as_volume`` needs to put the
    stack back into the original geometry.
    """
    if nib is None:
        raise ImportError('nibabel is required to read NIfTI files')
    image = nib.load(path)
    volume = np.squeeze(np.asanyarray(image.dataobj).astype(np.float32))
    if volume.ndim == 2:
        volume = volume[..., None]
    lo, hi = percentile_bounds(volume, pre.percentiles)
    n_slices = volume.shape[pre.slice_axis]

    slices, positions, indices = [], [], []
    threshold = lo + 0.10 * (hi - lo)
    for k in range(n_slices):
        raw = np.take(volume, k, axis=pre.slice_axis)
        if not keep_all_slices and pre.min_foreground_fraction > 0:
            if float(np.mean(raw > threshold)) < pre.min_foreground_fraction:
                continue
        if pre.norm_scope == 'slice':
            s_lo, s_hi = percentile_bounds(raw, pre.percentiles)
        else:
            s_lo, s_hi = lo, hi
        array = crop_or_pad_2d(rescale_to_pm1(raw, s_lo, s_hi),
                               pre.target_shape, pre.pad_value)
        slices.append(array)
        positions.append(k / (n_slices - 1) if n_slices > 1 else 0.5)
        indices.append(k)

    tensor = torch.from_numpy(np.stack(slices).astype(np.float32))[:, None]
    meta = {
        'affine': image.affine,
        'header': image.header,
        'original_shape': tuple(int(s) for s in volume.shape),
        'slice_axis': pre.slice_axis,
        'slice_indices': indices,
        'slice_pos': torch.tensor(positions, dtype=torch.float32),
        'lo': lo,
        'hi': hi,
        'path': path,
    }
    return tensor, meta


def save_slices_as_volume(slices: torch.Tensor, meta: Dict[str, object],
                          out_path: str, rescale_to_original: bool = True):
    """Undo crop/pad and write the stack back out as a NIfTI volume."""
    if nib is None:
        raise ImportError('nibabel is required to write NIfTI files')

    array = slices.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[:, 0]

    original_shape = meta['original_shape']
    slice_axis = int(meta['slice_axis'])
    plane_axes = [a for a in range(3) if a != slice_axis]
    plane_shape = (original_shape[plane_axes[0]], original_shape[plane_axes[1]])

    restored = np.stack([crop_or_pad_2d(sl, plane_shape, pad_value=-1.0)
                         for sl in array])

    volume = np.full(original_shape, -1.0, dtype=np.float32)
    for position, index in enumerate(meta['slice_indices']):
        selector: List[object] = [slice(None)] * 3
        selector[slice_axis] = index
        volume[tuple(selector)] = restored[position]

    if rescale_to_original:
        lo, hi = float(meta['lo']), float(meta['hi'])
        volume = (volume + 1.0) / 2.0 * (hi - lo) + lo

    nib.save(nib.Nifti1Image(volume, meta['affine']), out_path)
    return out_path
