"""Stage 1 -- train the 2D VQ-GAN.  Adapted from TATS / medicaldiffusion."""

import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from train.callbacks import ImageLogger
from train.get_dataset import get_dataset
from vq_gan_2d.model import VQGAN


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers,
                                  sampler=sampler, shuffle=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False,
                                num_workers=cfg.model.num_workers)

    bs = cfg.model.batch_size
    base_lr = cfg.model.lr
    ngpu = cfg.model.gpus
    accumulate = cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name,
            cfg.model.default_root_dir_postfix)
    print('Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} '
          '(num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)'.format(
              cfg.model.lr, accumulate, ngpu / 8, bs / 4, base_lr))

    model = VQGAN(cfg)

    tb_logger = TensorBoardLogger(save_dir=cfg.model.default_root_dir,
                                  name='lightning_logs')
    csv_logger = CSVLogger(save_dir=cfg.model.default_root_dir,
                           name='lightning_logs', version=tb_logger.version)
    ckpt_dir = os.path.join(tb_logger.log_dir, 'checkpoints')

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, monitor='val/recon_loss',
                        save_top_k=3, mode='min', filename='latest_checkpoint'),
        ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=3000,
                        save_top_k=-1,
                        filename='{epoch}-{step}-{train/recon_loss:.2f}'),
        ModelCheckpoint(dirpath=ckpt_dir, every_n_train_steps=10000,
                        save_top_k=-1,
                        filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'),
        ImageLogger(batch_frequency=750, max_images=4, clamp=True),
    ]

    # resume from the most recent checkpoint if there is one
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = 0
        for folder in os.listdir(base_dir):
            try:
                version_id = int(folder.split('_')[1])
            except (IndexError, ValueError):
                continue
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if log_folder:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.isdir(ckpt_folder):
                for fn in os.listdir(ckpt_folder):
                    if fn == 'latest_checkpoint.ckpt':
                        ckpt_file = 'latest_checkpoint_prev.ckpt'
                        os.replace(os.path.join(ckpt_folder, fn),
                                   os.path.join(ckpt_folder, ckpt_file))
            if ckpt_file:
                cfg.model.resume_from_checkpoint = os.path.join(ckpt_folder,
                                                                ckpt_file)
                print('will start from the recent ckpt %s'
                      % cfg.model.resume_from_checkpoint)

    accelerator = 'ddp' if cfg.model.gpus > 1 else None

    trainer = pl.Trainer(
        gpus=cfg.model.gpus,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()
