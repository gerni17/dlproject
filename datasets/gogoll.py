from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
# from sklearn.model_selection import train_test_split

# Agriculture Dataset ---------------------------------------------------------------------------
class GogollDataset(Dataset):
    def __init__(self, source_img_paths, segmentation_img_paths, target_img_paths, transform, phase="train"):
        self.source_img_paths = source_img_paths
        self.segmentation_img_paths = segmentation_img_paths
        self.target_img_paths = target_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.source_img_paths), len(self.segmentation_img_paths), len(self.target_img_paths)])

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        segmentation_img = Image.open(self.segmentation_img_paths[idx])
        target_img = Image.open(self.target_img_paths[idx])
        assert rgb_img.size == segmentation_img.size

        img, segmentation = self.transform(rgb_img, segmentation_img, self.phase)
        target, _ = self.transform(target_img, segmentation_img, self.phase)

        return { "id": idx, "source": img, "source_segmentation": segmentation, "target": target }


# Data Module
class GogollDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, domain, transform, batch_size):
        super(GogollDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.domain = domain
        self.batch_size = batch_size

    def prepare_data(self):

        self.train_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "train", "rgb", "*.png")
        )
        self.train_segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "train", "semseg", "*.png")
        )
        self.train_target_paths = glob.glob(
            os.path.join(self.data_dir, "other_domains", "train", self.domain, "*.jpg")
        )
        self.val_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "val", "rgb", "*.png")
        )
        self.val_segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "val", "semseg", "*.png")
        )
        self.val_target_paths = glob.glob(
            os.path.join(self.data_dir, "other_domains", "val", self.domain, "*.jpg")
        )
        self.test_rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "test", "rgb", "*.png")
        )
        self.test_segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "test", "semseg", "*.png")
        )
        self.test_target_paths = glob.glob(
            os.path.join(self.data_dir, "other_domains", "test", self.domain, "*.jpg")
        )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = GogollDataset(
            self.train_rgb_paths,
            self.train_segmentation_paths,
            self.train_target_paths,
            self.transform, "train"
        )
        self.val_dataset = GogollDataset(
            self.val_rgb_paths,
            self.val_segmentation_paths,
            self.val_target_paths,
            self.transform,
            "test"
        )
        self.test_dataset = GogollDataset(
            self.test_rgb_paths,
            self.test_segmentation_paths,
            self.test_target_paths,
            self.transform,
            "train"
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
