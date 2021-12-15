from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split

from utils.sanity import assert_matching_images

# GAM (Gian-Alex-Mugeeb) Dataset ---------------------------------------------------------------------------
class GamDataset(Dataset):
    def __init__(
        self,
        source_img_paths,
        segmentation_img_paths,
        transform,
        phase="train",
    ):
        self.source_img_paths = source_img_paths
        self.segmentation_img_paths = segmentation_img_paths
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
            ]
        )

    def __getitem__(self, idx):
        rgb_img = Image.open(self.source_img_paths[idx])
        segmentation_img = Image.open(self.segmentation_img_paths[idx])

        assert rgb_img.size == segmentation_img.size

        img, segmentation = self.transform(rgb_img, segmentation_img, self.phase)

        return {
            "id": idx,
            "source": img,
            "source_segmentation": segmentation,
        }


# Data Module
class GamDataModule(pl.LightningDataModule):
    def __init__(self, source_dir, transform, batch_size, split=True):
        super(GamDataModule, self).__init__()
        self.source_dir = source_dir
        self.transform = transform
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        self.rgb_paths = glob.glob(os.path.join(self.source_dir, "rgb", "*.png"))
        self.segmentation_paths = glob.glob(
            os.path.join(self.source_dir, "semseg", "*.png")
        )

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
        self.train_dataset = GamDataset(
            self.rgb_train, self.seg_train, self.transform, "train"
        )

        self.val_dataset = GamDataset(
            self.rgb_val, self.seg_val, self.transform, "train"
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
