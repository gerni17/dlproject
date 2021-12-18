from typing import Optional
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split

from utils.sanity import assert_matching_images


class GeneratedDataset(Dataset):
    def __init__(
        self,
        generator,
        dataset,
    ):
        self.generator = generator
        self.dataset = dataset
        
        self.raw_len = min(
            [len(self.dataset)]
        )

    def __len__(self):
        return self.raw_len

    def __getitem__(self, idx):
        d_item = self.dataset[idx]

        rgb_img = d_item['source']
        segmentation_img = d_item['source_segmentation']

        shape = rgb_img.shape

        with torch.no_grad():
            generated = self.generator(
                torch.reshape(rgb_img, (1, shape[0], shape[1], shape[2]))
            )
            generated = torch.reshape(
                generated, (generated.shape[1], generated.shape[2], generated.shape[3])
            )

        return {"id": idx, "source": generated, "source_segmentation": segmentation_img}


# Data Module
class GeneratedDataModule(pl.LightningDataModule):
    def __init__(
        self, generator, datamodule, batch_size
    ):
        super(GeneratedDataModule, self).__init__()
        self.generator = generator
        self.datamodule = datamodule
        self.batch_size = batch_size

    def prepare_data(self):
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.datamodule.setup(stage)
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = GeneratedDataset(
            self.generator,
            self.datamodule.train_dataloader().dataset,
        )

        self.val_dataset = GeneratedDataset(
            self.generator,
            self.datamodule.val_dataloader().dataset,
        )

        self.test_dataset = GeneratedDataset(
            self.generator,
            self.datamodule.test_dataloader().dataset,
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
