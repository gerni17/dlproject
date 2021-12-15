from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split


class MixedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = sum(len(s) for s in self.datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        c_len_start = 0
        c_len = 0

        for ds in self.datasets:
            c_len += len(ds)

            if idx < c_len:
                return ds[idx - c_len_start]

            c_len_start += len(ds)

        raise LookupError("Index was out of bounds for datasets")


# Data Module
class MixedDataModule(pl.LightningDataModule):
    def __init__(self, *datamodules, batch_size=4):
        super(MixedDataModule, self).__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size

    def prepare_data(self):
        for dm in self.datamodules:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        train_loaders = []
        val_loaders = []
        test_loaders = []

        for dm in self.datamodules:
            dm.setup()

            try:
                train_loaders.append(dm.train_dataloader())
            except:
                pass
            try:
                val_loaders.append(dm.val_dataloader())
            except:
                pass
            try:
                test_loaders.append(dm.test_dataloader())
            except:
                pass

        self.train_dataset = MixedDataset([x.dataset for x in train_loaders])
        self.val_dataset = MixedDataset([x.dataset for x in val_loaders])
        self.test_dataset = MixedDataset([x.dataset for x in test_loaders])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
