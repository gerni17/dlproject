import math
from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
import os, glob, random
from sklearn.model_selection import train_test_split
from datasets.labeled import LabeledDataset

from utils.sanity import assert_matching_images


# Data Module
class TestLabeledDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size):
        super(TestLabeledDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):
        self.rgb_paths = glob.glob(os.path.join(self.data_dir, "rgb", "*.png"))
        self.segmentation_paths = glob.glob(os.path.join(self.data_dir, "semseg", "*.png"))

    def setup(self, stage: Optional[str] = None):
        # Assign full dataset to all loaders
        self.full_dataset = LabeledDataset(self.rgb_paths, self.segmentation_paths, self.transform, "test")

    def test_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
