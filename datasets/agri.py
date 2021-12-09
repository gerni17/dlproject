from typing import Optional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import os, glob, random
from sklearn.model_selection import train_test_split

# Agriculture Dataset ---------------------------------------------------------------------------
class AgriDataset(Dataset):
    def __init__(self, source_img_paths, target_img_paths, transform, phase="train"):
        self.source_img_paths = source_img_paths
        self.target_img_paths = target_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.source_img_paths), len(self.target_img_paths)])

    def __getitem__(self, idx):
        source_img = Image.open(self.source_img_paths[idx])
        target_img = Image.open(self.target_img_paths[idx])

        # apply preprocessing transformations
        source_img = self.transform(source_img, self.phase)
        target_img = self.transform(target_img, self.phase)

        return {"source": source_img, "target": target_img}


# Data Module
class AgriDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, domain="domainA"):
        super(AgriDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.domain = domain

    def prepare_data(self):
        self.source_img_paths = glob.glob(
            os.path.join(self.data_dir, "exp", "train", "rgb", "*.png")
        )
        self.target_img_paths = glob.glob(
            os.path.join(self.data_dir, "other_domains", "train", self.domain, "*.jpg")
        )

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            random.shuffle(self.source_img_paths)
            random.shuffle(self.target_img_paths)

            self.train_dataset = AgriDataset(
                self.source_img_paths, self.target_img_paths, self.transform, "train"
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = AgriDataset(
                self.source_img_paths, self.target_img_paths, self.transform, "test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
