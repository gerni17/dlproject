from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split
import torch

from utils.sanity import assert_matching_images

def SegToOneHot(x):
    hot_map = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    encodings = torch.tensor(hot_map, dtype=torch.float32)
    x = x.cpu().clone()  # H x W
    x = encodings[x]  # H x W x 3
    x = x.permute(2, 0, 1)
    return x

# Agriculture Dataset ---------------------------------------------------------------------------
class GamDataset(Dataset):
    def __init__(
        self,
        source_img_paths,
        segmentation_img_paths,
        target_img_paths,
        transform,
        phase="train",
    ):
        self.source_img_paths = source_img_paths
        self.segmentation_img_paths = segmentation_img_paths
        self.target_img_paths = target_img_paths
        self.transform = transform
        self.phase = phase

        self.source_img_paths.sort()
        self.segmentation_img_paths.sort()
        assert_matching_images(self.source_img_paths, self.segmentation_img_paths)

    def __len__(self):
        return min(
            [
                len(self.source_img_paths),
                len(self.segmentation_img_paths),
                len(self.target_img_paths),
            ]
        )

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        segmentation_img = Image.open(self.segmentation_img_paths[idx])
        target_img = Image.open(self.target_img_paths[idx])
        assert rgb_img.size == segmentation_img.size

        img, segmentation = self.transform(rgb_img, segmentation_img, self.phase)
        target, _ = self.transform(target_img, segmentation_img, self.phase)

        return {
            "id": idx,
            "source": img,
            "source_segmentation": SegToOneHot(segmentation),
            "target": target,
        }


# Data Module
class GamDataModule(pl.LightningDataModule):
    def __init__(self, source_dir, target_dir, transform, batch_size, split=True):
        super(GamDataModule, self).__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        self.rgb_paths = glob.glob(os.path.join(self.source_dir, "rgb", "*.png"))
        self.segmentation_paths = glob.glob(
            os.path.join(self.source_dir, "semseg", "*.png")
        )
        self.target_paths = glob.glob(
            os.path.join(self.target_dir, "*.jpg")
        ) + glob.glob(
            os.path.join(self.target_dir, "*.png")
        )

        if self.split:
            (
                self.rgb_train,
                self.rgb_val,
                self.seg_train,
                self.seg_val,
            ) = train_test_split(self.rgb_paths, self.segmentation_paths, test_size=0.2)
            self.target_train, self.target_val = train_test_split(
                self.target_paths, test_size=0.2
            )
        else:
            self.rgb_train = self.rgb_paths
            self.rgb_val = []
            self.seg_train = self.segmentation_paths
            self.seg_val = []
            self.target_train = self.target_paths
            self.target_val = []

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = GamDataset(
            self.rgb_train, self.seg_train, self.target_train, self.transform, "train"
        )

        self.val_dataset = GamDataset(
            self.rgb_val, self.seg_val, self.target_val, self.transform, "train"
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
