import math
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import os, glob, random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

# Source domain dataset
class SourceDataset(Dataset):
    def __init__(self, source_img_paths, segmentation_img_paths, transform, phase="train"):
        self.source_img_paths = source_img_paths
        self.segmentation_img_paths = segmentation_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.source_img_paths), len(self.segmentation_img_paths)])

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        segmentation_img = Image.open(self.segmentation_img_paths[idx])
        assert rgb_img.size == segmentation_img.size

        img, segmentation = self.transform(rgb_img, segmentation_img, self.phase)

        return { "id": idx, "source": img, "source_segmentation": segmentation }


# Data Module
class SourceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, n_splits=5, active_split=0):
        super(SourceDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.rgb_train_splits = []
        self.seg_train_splits = []
        self.rgb_test_splits = []
        self.seg_test_splits = []
        self.train_datasets = []
        self.test_datasets = []
        self.n_splits = n_splits
        self.active_split = active_split
        self.cv_splitter = KFold(n_splits=self.n_splits, random_state=None, shuffle=False)

    def set_active_split(self, split_index):
        assert split_index >= 0 and split_index  < self.n_splits
        self.active_split = split_index

    def prepare_data(self):
        self.rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "rgb", "*.png")
        )
        self.segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "semseg", "*.png")
        )

        rgb_paths_np = np.array(self.rgb_paths)
        seg_paths_np = np.array(self.segmentation_paths)

        for train_index, test_index in self.cv_splitter.split(self.rgb_paths):
            X_train, X_test = rgb_paths_np[train_index], rgb_paths_np[test_index]
            y_train, y_test = seg_paths_np[train_index], seg_paths_np[test_index]
            self.rgb_train_splits.append(X_train)
            self.rgb_test_splits.append(X_test)
            self.seg_train_splits.append(y_train)
            self.seg_test_splits.append(y_test)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders

        for i in range(self.n_splits):
            train_dataset = SourceDataset(
                self.rgb_train_splits[i].tolist(),
                self.seg_train_splits[i].tolist(),
                self.transform,
                "train",
            )
            self.train_datasets.append(train_dataset)

        for i in range(self.n_splits):
            test_dataset = SourceDataset(
                self.rgb_test_splits[i].tolist(),
                self.seg_test_splits[i].tolist(),
                self.transform,
                "train",
            )
            self.test_datasets.append(test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_datasets[self.active_split],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_datasets[self.active_split],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_datasets[self.active_split],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
