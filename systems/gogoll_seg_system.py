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
from torch.optim.lr_scheduler import LambdaLR


class GogollSegSystem(pl.LightningModule):
    def __init__(self, net, cfg):  # segmentation network
        super(GogollSegSystem, self).__init__()
        self.net = net
        self.cfg = cfg
        self.step = 0

        self.semseg_loss = torch.nn.CrossEntropyLoss()
        # self.semseg_loss = torchmetrics.IoU(3)
        self.losses = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.seg_lr, betas=(0.5, 0.999),)
        if self.cfg.sched:
            sched=LambdaLR(
                optimizer,
                lambda ep: max(1e-6, (1 - ep / self.cfg.num_epochs_seg) ** self.cfg.lr_scheduler_power))
            return [optimizer], [sched]
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        source_img, segmentation_img = (
            batch["source"],
            batch["source_segmentation"],
        )

        y_seg = self.net(source_img)

        Seg_loss = self.semseg_loss(y_seg, segmentation_img)

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
