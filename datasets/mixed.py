from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split


# Mixed Dataset (Generated images + true images) ------------------------------------------------------
class MixedDataset(Dataset):
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
class MixedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, generated_dir, transform, batch_size, val_size = 0.2, test_size = 0.1):
        super(MixedDataModule, self).__init__()
        self.data_dir = data_dir
        self.generated_dir = generated_dir
        self.transform = transform
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

    def prepare_data(self):
        self.rgb_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "rgb", "*.png")
        ) + glob.glob(
            os.path.join(self.generated_dir, "rgb", "*.png")
        )

        self.segmentation_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "semseg", "*.png")
        ) + glob.glob(
            os.path.join(self.generated_dir, "semseg", "*.png")
        )

        if self.val_size + self.test_size >= 1:
            raise ValueError("Val size + test size must be smaller than 1")

        self.train_rgb, self.train_seg, self.val_rgb, self.val_seg = train_test_split(self.rgb_paths, self.segmentation_paths, self.val_size + self.test_size)
        self.val_rgb, self.val_seg, self.test_rgb, self.test_seg = train_test_split(self.val_rgb, self.val_seg, self.test_size/(self.val_size + self.test_size))

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = MixedDataset(
            self.train_rgb,
            self.train_seg,
            self.transform, "train"
        )
        self.val_dataset = MixedDataset(
            self.val_rgb,
            self.val_seg,
            self.transform, "train"
        )
        self.test_dataset = MixedDataset(
            self.test_rgb,
            self.test_seg,
            self.transform, "test"
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
