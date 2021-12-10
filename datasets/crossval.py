from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, SubsetRandomSampler
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

from datasets.mixed import MixedDataset

# Data Module
class CrossValidationDataModule(pl.LightningDataModule):
    def __init__(self, datamodule, batch_size=4, n_splits=5, active_split=0):
        super(CrossValidationDataModule, self).__init__()
        self.datamodule = datamodule
        self.batch_size = batch_size
        self.train_subsamplers = []
        self.val_subsamplers = []
        self.n_splits = n_splits
        self.active_split = active_split
        self.cv_splitter = KFold(
            n_splits=self.n_splits, random_state=None, shuffle=False
        )

    def set_active_split(self, split_index):
        assert split_index >= 0 and split_index < self.n_splits
        self.active_split = split_index

    def prepare_data(self):
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.train_subsamplers = []
        self.val_subsamplers = []

        self.datamodule.setup()

        # for crossvalidation we mix (and use both) train and val dataset (instead of a static split with train_dataset and val_dataset)
        self.train_val_dataset = MixedDataset([
            self.datamodule.train_dataset,
            self.datamodule.val_dataset
        ])

        # for test we keep the dataset that is unseen during training
        self.test_dataset = self.datamodule.test_dataset

        for train_index, test_index in self.cv_splitter.split(self.train_val_dataset):
            train_subsampler = SubsetRandomSampler(train_index)
            val_subsampler = SubsetRandomSampler(test_index)
            self.train_subsamplers.append(train_subsampler)
            self.val_subsamplers.append(val_subsampler)

    def train_dataloader(self):
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=self.train_subsamplers[self.active_split],
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=self.val_subsamplers[self.active_split],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
        )
