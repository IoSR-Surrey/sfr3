from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from sr3_modules.diffusion import GaussianDiffusion
from torch.utils.data import DataLoader
from data import SoundFieldDataset
import pytorch_lightning as pl
from model import SR3UNet
import argparse
import torch
import csv
import os


class SR3DiffusionModule(pl.LightningModule):
    def __init__(self, hr_res, lr, max_epochs, warmup_epochs):
        super().__init__()

        self.save_hyperparameters()

        model = SR3UNet(grid_res=hr_res)
        self.diffusion = GaussianDiffusion(denoise_fn=model, image_size=hr_res, channels=2)
        self.diffusion.set_new_noise_schedule({'schedule': 'cosine', 'n_timestep': 1000,
                                               'linear_start': 1e-4, 'linear_end': 2e-2}, device=torch.device('cpu'))
        self.diffusion.set_loss(device=torch.device('cpu'))

    def training_step(self, batch, batch_idx):
        gt_hr, lr_low, freq = batch
        lr_cond = self.diffusion.denoise_fn.upsample_lr(lr_low)
        loss = self.diffusion({'HR': gt_hr, 'SR': lr_cond, 'freq': freq.float()})

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        gt_hr, lr_low, freq = batch
        lr_cond = self.diffusion.denoise_fn.upsample_lr(lr_low)
        loss = self.diffusion({'HR': gt_hr, 'SR': lr_cond, 'freq': freq.float()})
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion.denoise_fn.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

        # ramps linearly for warmup_epochs then follows a cosine decay to 0 over max_epochs
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs,
                                                  max_epochs=self.hparams.max_epochs, warmup_start_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }


class SoundFieldDataModule(pl.LightningDataModule):
    def __init__(self, train_metadata_path, val_metadata_path, batch_size, num_workers):
        super().__init__()
        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_res = None

    def setup(self, stage=None):
        self.train_ds = SoundFieldDataset(path=self.train_metadata_path)
        self.val_ds = SoundFieldDataset(path=self.val_metadata_path)
        self.hr_res = self.train_ds.hr_res

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=max(1, self.num_workers // 2), pin_memory=True)


class CSVHistoryLogger(pl.Callback):
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "training_history.csv")
        self._header_written = False

    def on_fit_start(self, trainer, pl_module):
        if trainer.ckpt_path is None:
            if os.path.exists(self.path):
                os.remove(self.path)
            self._header_written = False
        else:
            self._header_written = os.path.exists(self.path)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        row = {
            "epoch": trainer.current_epoch,
            "train_loss": float(metrics.get("train_loss", float("nan"))),
            "val_loss": float(metrics.get("val_loss", float("nan"))),
            "lr": trainer.optimizers[0].param_groups[0]["lr"],
        }
        mode = 'w' if not self._header_written else 'a'
        with open(self.path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', dest='suffix', type=str, default="4to32_500_(2000rooms)")
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=50)
    args = parser.parse_args()

    # seeds torch, CUDA, numpy, and random
    pl.seed_everything(args.seed, workers=True)

    s = args.suffix.strip()
    out_dir = f"checkpoints_{s}" if s else "checkpoints"
    train_metadata = f"dataset_{s}/train/metadata.json" if s else "dataset/train/metadata.json"
    val_metadata = f"dataset_{s}/val/metadata.json" if s else "dataset/val/metadata.json"

    datamodule = SoundFieldDataModule(train_metadata_path=train_metadata, val_metadata_path=val_metadata,
                                      batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup()

    model = SR3DiffusionModule(hr_res=datamodule.hr_res, lr=args.lr, max_epochs=args.max_epochs,
                               warmup_epochs=args.warmup_epochs)

    # saves the single best checkpoint by val_loss and last.ckpt every epoch
    checkpoint_best = ModelCheckpoint(dirpath=out_dir, filename="bestval_model", monitor="val_loss", save_top_k=1,
                                      mode="min", save_last=True)
    checkpoint_periodic = ModelCheckpoint(dirpath=out_dir, filename="ckpt_{epoch}", every_n_epochs=args.save_every,
                                          save_top_k=-1)

    # writes metrics.csv (epoch, train_loss, val_loss, lr)
    csv_logger = CSVHistoryLogger(save_dir=out_dir)

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="auto", devices=1, precision="16",
                         gradient_clip_val=1.0, gradient_clip_algorithm="norm",
                         callbacks=[checkpoint_best, checkpoint_periodic, csv_logger], logger=False)

    resume_path = os.path.join(out_dir, "last.ckpt") if args.resume else None
    if args.resume and not os.path.exists(resume_path):
        resume_path = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_path)
    print(f"\nTraining done!")