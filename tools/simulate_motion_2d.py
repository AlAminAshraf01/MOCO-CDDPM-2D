"""Retrospective 2D in-plane rigid-motion simulation in k-space.

The 3D repository assumed you already had ``target/`` and ``corrupted/``
folders.  For a *2D* model the corruption has to be generated the way a 2D
multi-slice acquisition actually produces it, which is also the main physical
argument for going 2D at low field:

    In a 2D multi-slice sequence, phase encoding happens **within the slice**.
    Rigid head motion during the acquisition therefore imprints ghosting and
    blurring along the in-plane phase-encode direction of that slice -- the
    artefact lives in the same plane the network sees.

The simulation follows the standard retrospective recipe: the object occupies a
different rigid pose during each k-space segment, so for each motion state we
compute the k-space of the transformed image and copy that state's phase-encode
lines into the corrupted k-space.  Rotation is applied in image space (the
Fourier rotation theorem makes the two equivalent) and translation as an exact
linear phase ramp.

Usage::

    python tools/simulate_motion_2d.py --input D:\\data\\clean_volumes \\
        --output D:\\data\\cddpm_2d_training --pattern sudden --severity moderate

The output directory gets ``target/`` (a copy of the clean volume) and
``corrupted/`` with matching file names, which is exactly what
``dataset.use_paired=true`` expects.
"""

import argparse
import os
import shutil

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

SEVERITY = {
    #                    max rotation (deg), max translation (pixels)
    'mild':     (2.0, 2.0),
    'moderate': (5.0, 5.0),
    'severe':   (10.0, 10.0),
}


def _rotate(image, angle_deg):
    if abs(angle_deg) < 1e-4:
        return image
    from scipy import ndimage
    return ndimage.rotate(image, angle_deg, reshape=False, order=1,
                          mode='constant', cval=0.0)


def _translation_phase(shape, shift):
    """Exact sub-pixel translation as a linear phase ramp in k-space."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    return np.exp(-2j * np.pi * (ky * shift[0] + kx * shift[1]))


def _motion_states(n_states, pattern, max_rot, max_trans, rng):
    """Return ``n_states`` (angle_deg, (dy, dx)) poses."""
    if pattern == 'sudden':
        # one abrupt movement partway through the acquisition
        switch = rng.integers(max(1, n_states // 4), max(2, 3 * n_states // 4))
        angle = rng.uniform(-max_rot, max_rot)
        shift = rng.uniform(-max_trans, max_trans, size=2)
        return [(0.0, np.zeros(2)) if i < switch else (angle, shift)
                for i in range(n_states)]

    if pattern == 'random_walk':
        angles, shifts = [], []
        angle, shift = 0.0, np.zeros(2)
        for _ in range(n_states):
            angle = np.clip(angle + rng.normal(0, max_rot / 3.0),
                            -max_rot, max_rot)
            shift = np.clip(shift + rng.normal(0, max_trans / 3.0, size=2),
                            -max_trans, max_trans)
            angles.append(float(angle))
            shifts.append(shift.copy())
        return list(zip(angles, shifts))

    if pattern == 'periodic':
        phase = rng.uniform(0, 2 * np.pi)
        cycles = rng.uniform(1.0, 3.0)
        out = []
        for i in range(n_states):
            arg = 2 * np.pi * cycles * i / max(1, n_states - 1) + phase
            out.append((max_rot * np.sin(arg),
                        max_trans * np.array([np.sin(arg), np.cos(arg)])))
        return out

    raise ValueError(f'Unknown pattern {pattern}')


def corrupt_slice(image, n_states=8, pattern='sudden', severity='moderate',
                  pe_axis=0, rng=None, center_fraction=0.0):
    """Corrupt one 2D slice.  ``pe_axis`` 0 = rows, 1 = columns."""
    rng = rng or np.random.default_rng()
    max_rot, max_trans = SEVERITY[severity]
    states = _motion_states(n_states, pattern, max_rot, max_trans, rng)

    n_lines = image.shape[pe_axis]
    # interleave assignment: contiguous segments, i.e. a sequential PE order
    bounds = np.linspace(0, n_lines, n_states + 1).astype(int)

    corrupted = np.zeros(image.shape, dtype=np.complex128)
    for state_index, (angle, shift) in enumerate(states):
        moved = _rotate(image, angle)
        kspace = np.fft.fft2(moved)
        kspace = kspace * _translation_phase(image.shape, shift)

        lo, hi = bounds[state_index], bounds[state_index + 1]
        if lo == hi:
            continue
        selector = [slice(None), slice(None)]
        selector[pe_axis] = slice(lo, hi)
        # fftfreq ordering: lines are indexed in the shifted (natural) order
        corrupted_view = tuple(selector)
        corrupted[corrupted_view] = np.fft.fftshift(
            kspace, axes=pe_axis)[corrupted_view]

    corrupted = np.fft.ifftshift(corrupted, axes=pe_axis)

    if center_fraction > 0:
        # optionally keep the central k-space lines motion free (a short
        # centric-ordered readout barely moves)
        clean_k = np.fft.fftshift(np.fft.fft2(image), axes=pe_axis)
        shifted = np.fft.fftshift(corrupted, axes=pe_axis)
        half = int(n_lines * center_fraction / 2)
        mid = n_lines // 2
        selector = [slice(None), slice(None)]
        selector[pe_axis] = slice(mid - half, mid + half)
        shifted[tuple(selector)] = clean_k[tuple(selector)]
        corrupted = np.fft.ifftshift(shifted, axes=pe_axis)

    return np.abs(np.fft.ifft2(corrupted)).astype(np.float32)


def corrupt_volume(volume, slice_axis=2, **kwargs):
    out = np.empty_like(volume, dtype=np.float32)
    for k in range(volume.shape[slice_axis]):
        selector = [slice(None)] * volume.ndim
        selector[slice_axis] = k
        sl = volume[tuple(selector)]
        out[tuple(selector)] = corrupt_slice(sl.astype(np.float64), **kwargs)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Generate paired motion-free / motion-corrupted 2D data')
    parser.add_argument('--input', required=True,
                        help='folder of clean NIfTI volumes')
    parser.add_argument('--output', required=True,
                        help='destination root (gets target/ and corrupted/)')
    parser.add_argument('--pattern', default='sudden',
                        choices=['sudden', 'random_walk', 'periodic'])
    parser.add_argument('--severity', default='moderate',
                        choices=list(SEVERITY))
    parser.add_argument('--n-states', type=int, default=8,
                        help='number of distinct poses per slice')
    parser.add_argument('--pe-axis', type=int, default=0, choices=[0, 1],
                        help='in-plane phase-encode axis of the slice')
    parser.add_argument('--slice-axis', type=int, default=2)
    parser.add_argument('--center-fraction', type=float, default=0.0,
                        help='fraction of central PE lines left motion free')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-slice-motion', action='store_true',
                        help='draw an independent motion trace for each slice '
                             '(otherwise one trace is reused for the volume)')
    args = parser.parse_args()

    if nib is None:
        raise SystemExit('nibabel is required: pip install nibabel')

    target_dir = os.path.join(args.output, 'target')
    cond_dir = os.path.join(args.output, 'corrupted')
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cond_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(args.input)
                   if n.lower().endswith(('.nii', '.nii.gz')))
    if not names:
        raise SystemExit(f'No NIfTI files found in {args.input}')

    for index, name in enumerate(names):
        src = os.path.join(args.input, name)
        image = nib.load(src)
        volume = np.squeeze(np.asanyarray(image.dataobj).astype(np.float64))
        if volume.ndim != 3:
            print(f'  skipping {name}: expected a 3D volume, got {volume.shape}')
            continue

        seed = args.seed + index
        rng = np.random.default_rng(seed)
        corrupted = np.empty_like(volume, dtype=np.float32)
        for k in range(volume.shape[args.slice_axis]):
            selector = [slice(None)] * 3
            selector[args.slice_axis] = k
            # per-slice: an independent trace per slice; otherwise the same
            # trace is replayed for every slice of the volume
            slice_seed = seed * 1000 + k if args.per_slice_motion else seed
            slice_rng = np.random.default_rng(slice_seed)
            corrupted[tuple(selector)] = corrupt_slice(
                volume[tuple(selector)], n_states=args.n_states,
                pattern=args.pattern, severity=args.severity,
                pe_axis=args.pe_axis, rng=slice_rng,
                center_fraction=args.center_fraction)

        shutil.copyfile(src, os.path.join(target_dir, name))
        nib.save(nib.Nifti1Image(corrupted, image.affine),
                 os.path.join(cond_dir, name))
        print(f'[{index + 1}/{len(names)}] {name}')

    print(f'done -- paired data in {args.output}')


if __name__ == '__main__':
    main()
