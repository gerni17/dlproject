from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

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
class MixedCrossValDataModule(pl.LightningDataModule):
    def __init__(self, *datamodules, batch_size=4, n_splits=5, active_split=0):
        super(MixedCrossValDataModule, self).__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size
        self.train_subsamplers = []
        self.test_subsamplers = []
        self.n_splits = n_splits
        self.active_split = active_split
        self.cv_splitter = KFold(n_splits=self.n_splits, random_state=None, shuffle=False)

    def get_datamoduels(self):
        return self.datamodules

    def set_active_split(self, split_index):
        assert split_index >= 0 and split_index  < self.n_splits
        self.active_split = split_index

    def prepare_data(self):
        for dm in self.datamodules:
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        all_datasets = []

        for dm in self.datamodules:
            dm.setup()
            all_datasets.append(dm.full_dataset)

        self.concatenated_dataset = ConcatDataset(all_datasets)
        for train_index, test_index in self.cv_splitter.split(self.concatenated_dataset):
            train_subsampler = SubsetRandomSampler(train_index)
            test_subsampler = SubsetRandomSampler(test_index)
            self.train_subsamplers.append(train_subsampler)
            self.test_subsamplers.append(test_subsampler)


    def train_dataloader(self):
        return DataLoader(
            self.concatenated_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=self.train_subsamplers[self.active_split]
        )

    def val_dataloader(self):
        return DataLoader(
            self.concatenated_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=self.test_subsamplers[self.active_split]
        )

    def test_dataloader(self):
        return DataLoader(
            self.concatenated_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=self.test_subsamplers[self.active_split]
        )
