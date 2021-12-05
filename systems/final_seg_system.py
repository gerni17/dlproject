import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import chain
import torch
from torchvision.utils import make_grid
from torch import nn, optim
import pytorch_lightning as pl
from utils.metrics import MetricsSemseg
from torchmetrics import IoU


class FinalSegSystem(pl.LightningModule):
    def __init__(self, net, lr=0.0002):  # segmentation network
        super(FinalSegSystem, self).__init__()
        self.net = net
        self.lr = lr
        self.step = 0

        # self.semseg_loss = torch.nn.CrossEntropyLoss()
        self.iou_metric = IoU(num_classes=3)
        self.losses = []
        names=["crop","weed", "soil"]
        self.metrics_semseg = MetricsSemseg(3, names)

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
    
    def validation_step(self, batch, batch_idx):
        source_img, segmentation_img = (batch["source"], batch["source_segmentation"])
        y_hat=self.net(source_img)
        loss_val_semseg = self.semseg_loss(y_hat, segmentation_img)

        y_hat_semseg_lbl = y_hat.argmax(dim=1)
        self.metrics_semseg.update_batch(y_hat_semseg_lbl, segmentation_img)

        self.log_dict({
                'loss_val/semseg': loss_val_semseg,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metric_semseg = metrics_semseg['mean_iou']

        scalar_logs = {
            'metrics_summary/semseg': metric_semseg,
        }
        scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        source_img, segmentation_img = (batch["source"], batch["source_segmentation"])

        y_seg = self.net(source_img)

        jaccard_index = self.iou_metric(y_seg, segmentation_img.int())

        logs = {
            "IOU Metric": jaccard_index,
        }

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def segment(self, inputs):
        return self.net(inputs)
