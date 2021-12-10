import math
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import os, glob, random
from sklearn.model_selection import train_test_split

from utils.sanity import assert_matching_images

# Source domain dataset
class SourceDataset(Dataset):
    def __init__(
        self,
        source_img_paths,
        segmentation_img_paths,
        transform,
        phase="train",
        max_imgs=200,
    ):
        self.source_img_paths = source_img_paths
        self.segmentation_img_paths = segmentation_img_paths
        self.transform = transform
        self.phase = phase
        self.max_imgs = max_imgs

        self.source_img_paths.sort()
        self.segmentation_img_paths.sort()
        assert_matching_images(self.source_img_paths, self.segmentation_img_paths)

    def __len__(self):
        return min(
            [
                len(self.source_img_paths),
                len(self.segmentation_img_paths),
                self.max_imgs,
            ]
        )

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        segmentation_img = Image.open(self.segmentation_img_paths[idx])
        assert rgb_img.size == segmentation_img.size

        img, segmentation = self.transform(rgb_img, segmentation_img, self.phase)

        return {"id": idx, "source": img, "source_segmentation": segmentation}


# Data Module
class SourceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, split=True, max_imgs=200):
        super(SourceDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.split = split
        self.max_imgs = max_imgs

    def prepare_data(self):
        self.rgb_paths = glob.glob(os.path.join(self.data_dir, "rgb", "*.png"))
        self.segmentation_paths = glob.glob(os.path.join(self.data_dir, "semseg", "*.png"))

        if self.split:
            (
                self.rgb_train,
                self.rgb_val,
                self.seg_train,
                self.seg_val,
            ) = train_test_split(self.rgb_paths, self.segmentation_paths, test_size=0.2)
        else:
            self.rgb_train = self.rgb_paths
            self.rgb_val = []
            self.seg_train = self.segmentation_paths
            self.seg_val = []

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = SourceDataset(self.rgb_train, self.seg_train, self.transform, "train", self.max_imgs)
        self.val_dataset = SourceDataset(
            self.rgb_val,
            self.seg_val,
            self.transform,
            "test",
            self.max_imgs,
        )

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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
