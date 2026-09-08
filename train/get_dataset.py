"""Dataset factory -- 2D version of ``train/get_dataset.py``.

The train/val split is always done on *files*, never on slices, so that no
subject appears on both sides of the split.  That matters much more in 2D than
in 3D: neighbouring slices of the same subject are near-duplicates and a
slice-level split silently inflates validation scores.
"""

from dataset.default import (Augment2D, DEFAULT2DDataset,
                             DEFAULT2DPairedDataset, Preprocess2D,
                             split_sequence_deterministically)


def get_dataset(cfg):
    if cfg.dataset.name != 'DEFAULT':
        raise ValueError('Only the DEFAULT dataset is available in this repository')

    pre = Preprocess2D.from_cfg(cfg.dataset)
    aug = Augment2D.from_cfg(cfg.dataset)

    use_paired = bool(getattr(cfg.dataset, 'use_paired', False))
    enable_split = bool(getattr(cfg.dataset, 'enable_split', False))
    val_fraction = float(getattr(cfg.dataset, 'val_fraction', 0.1))
    split_seed = int(getattr(cfg.dataset, 'split_seed', 42))
    root_dir = cfg.dataset.root_dir

    if use_paired:
        target_subdir = getattr(cfg.dataset, 'target_subdir', 'target')
        condition_subdir = getattr(cfg.dataset, 'condition_subdir', 'corrupted')

        def build(augment, file_names=None):
            return DEFAULT2DPairedDataset(
                root_dir=root_dir, target_subdir=target_subdir,
                condition_subdir=condition_subdir, pre=pre, augment=augment,
                aug=aug, file_names=file_names)

        if enable_split:
            probe = DEFAULT2DPairedDataset(
                root_dir=root_dir, target_subdir=target_subdir,
                condition_subdir=condition_subdir, pre=pre, augment=False,
                aug=aug)
            train_names, val_names = split_sequence_deterministically(
                probe.file_names, val_fraction=val_fraction, seed=split_seed)
            train_dataset = build(True, train_names)
            val_dataset = build(False, val_names) if val_names else build(False)
        else:
            train_dataset = build(True)
            val_dataset = build(False)
    else:
        def build(augment, file_paths=None):
            return DEFAULT2DDataset(root_dir=root_dir, pre=pre,
                                    augment=augment, aug=aug,
                                    file_paths=file_paths)

        if enable_split:
            probe = DEFAULT2DDataset(root_dir=root_dir, pre=pre, augment=False,
                                     aug=aug)
            train_paths, val_paths = split_sequence_deterministically(
                probe.file_paths, val_fraction=val_fraction, seed=split_seed)
            train_dataset = build(True, train_paths)
            val_dataset = build(False, val_paths) if val_paths else build(False)
        else:
            train_dataset = build(True)
            val_dataset = build(False)

    sampler = None
    return train_dataset, val_dataset, sampler
