from src.dataloader.dataset import G2Dataset
from src.dataloader.dataset_utils import load_augs
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import pandas as pd
import hydra
from omegaconf import DictConfig


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        df = pd.read_csv(self.cfg.datamodule.df_path)
        self.train_data = df[df.kfold != self.cfg.datamodule.fold]
        self.valid_data = df[df.kfold == self.cfg.datamodule.fold]
        self.train_transforms = load_augs(self.cfg["augmentation"]["train"]["augs"])
        self.val_transforms = load_augs(self.cfg["augmentation"]["valid"]["augs"])
        self.num_workers = self.cfg.datamodule.num_workers
        self.pin_memory = self.cfg.datamodulepin_memory
        self.train_batch_size = self.cfg.datamoduletrain_batch_size
        self.val_batch_size = self.cfg.datamoduleval_batch_size

    def setup(self, stage=None):
        self.train_dataset = G2Dataset(
            images_filepaths=self.train_data["image_path"].values,
            targets=self.train_data["target"].values,
            transform=hydra.utils.instantiate(self.train_transforms),
        )

        self.val_dataset = G2Dataset(
            images_filepaths=self.valid_data["image_path"].values,
            targets=self.valid_data["target"].values,
            transform=hydra.utils.instantiate(self.val_transforms),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
