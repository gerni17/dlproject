import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import chain
import torch
import torchmetrics
from torchvision.utils import make_grid
from torch import nn, optim
import pytorch_lightning as pl


class FinalSegSystem(pl.LightningModule):
    def __init__(self, net, lr=0.0002):  # segmentation network
        super(FinalSegSystem, self).__init__()
        self.net = net
        self.lr = lr
        self.step = 0

        self.semseg_loss = torch.nn.CrossEntropyLoss()
        # self.semseg_loss = torchmetrics.IoU(3)
        self.losses = []

    def configure_optimizers(self):
        self.global_optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
        )

        return [
            self.global_optimizer,
        ], []

    def training_step(self, batch, batch_idx):
        source_img, segmentation_img = (batch["source"], batch["source_segmentation"])

        y_seg = self.net(source_img)

        Seg_loss = self.semseg_loss(y_seg, segmentation_img)
        # Seg_loss = 1 - self.semseg_loss(y_seg, segmentation_img)
        # Seg_loss.requires_grad = True

        logs = {
            "loss": Seg_loss,
        }

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": Seg_loss}

    def training_epoch_end(self, outputs):
        self.step += 1

        self.losses.append(outputs[0]["loss"])

        return None

    def segment(self, inputs):
        return self.net(inputs)
