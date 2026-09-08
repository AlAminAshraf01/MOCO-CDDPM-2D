"""Lightning callbacks -- 2D version.

``VideoLogger`` from the 3D repo is gone: there is no depth axis to animate.
``ImageLogger`` now writes side-by-side input/reconstruction PNGs.
"""

import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback

try:  # pytorch-lightning >= 1.7 moved this
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:  # pragma: no cover
    from pytorch_lightning.utilities.rank_zero import rank_zero_only


def _resolve_save_dir(trainer, pl_module):
    cfg = getattr(pl_module, 'cfg', None)
    if cfg is not None:
        model_cfg = getattr(cfg, 'model', None)
        if model_cfg is not None:
            base_dir = getattr(model_cfg, 'default_root_dir', None)
            if base_dir:
                return str(base_dir)
    return getattr(trainer, 'log_dir', None) or getattr(
        trainer, 'default_root_dir', None)


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True,
                 increase_log_steps=True):
        super().__init__()
        self.batch_freq = max(1, batch_frequency)
        self.max_images = max_images
        self.log_steps = [2 ** n for n in
                          range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            image = images[k]
            if self.clamp:
                image = image.clamp(-1., 1.)
            image = (image + 1.0) * 127.5
            grid = torchvision.utils.make_grid(image, nrow=4)
            grid = grid.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
            if grid.shape[-1] == 1:
                grid = grid[..., 0]
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx)
            Image.fromarray(grid).save(os.path.join(root, filename))

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        if not (self.check_frequency(batch_idx)
                and hasattr(pl_module, "log_images")
                and callable(pl_module.log_images)
                and self.max_images > 0):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split)

        for k in images:
            n = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:n]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()

        save_dir = _resolve_save_dir(trainer, pl_module)
        if save_dir:
            self.log_local(save_dir, split, images, pl_module.global_step,
                           pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                           dataloader_idx=0):
        self.log_img(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        self.log_img(trainer, pl_module, batch, batch_idx, split="val")
