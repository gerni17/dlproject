from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random

from datasets.splitter import Splitter

# Agriculture Source Domain Dataset ---------------------------------------------------------------------------
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
    def __init__(self, data_dir, transform, batch_size):
        super(SourceDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):
        self.rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "rgb", "*.png")
        )
        self.segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "semseg", "*.png")
        )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.dataset = SourceDataset(
            self.rgb_paths,
            self.segmentation_paths,
            self.transform, "test"
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
