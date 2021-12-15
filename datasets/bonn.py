from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random

# from sklearn.model_selection import train_test_split

# Agriculture Dataset ---------------------------------------------------------------------------
class BonnDataset(Dataset):
    def __init__(self, source_img_paths, target_img_paths, transform, phase="train"):
        self.source_img_paths = source_img_paths
        self.target_img_paths = target_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.source_img_paths), len(self.target_img_paths)])

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        target_img = Image.open(self.target_img_paths[idx])
        assert rgb_img.size == target_img.size

        img, target = self.transform(rgb_img, target_img, self.phase)

        return {"id": idx, "rgb": img, "label": target}


# Data Module
class BonnDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size):
        super(BonnDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):

        self.train_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "train", "rgb", "*.png")
        )
        self.train_label_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "train", "semseg", "*.png")
        )
        self.val_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "val", "rgb", "*.png")
        )
        self.val_label_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "val", "semseg", "*.png")
        )
        self.test_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "test", "rgb", "*.png")
        )
        self.test_label_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "test", "semseg", "*.png")
        )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BonnDataset(
            self.train_rgb_paths, self.train_label_paths, self.transform, "train"
        )
        self.val_dataset = BonnDataset(
            self.val_rgb_paths, self.val_label_paths, self.transform, "test"
        )
        self.test_dataset = BonnDataset(
            self.test_rgb_paths, self.test_label_paths, self.transform, "train"
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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
