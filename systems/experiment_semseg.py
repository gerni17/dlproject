import os
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch import nn, optim
from utils.metrics import MetricsSemseg
from torchmetrics import IoU
from torch.optim.lr_scheduler import LambdaLR



class Semseg(pl.LightningModule):
    def __init__(self, net,cfg):
        super(Semseg, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.net = net

        self.loss_semseg = torch.nn.CrossEntropyLoss()
        names = ["soil", "crop", "weed"]
        self.metrics_semseg = MetricsSemseg(3, names)
        self.iou_metric = IoU(num_classes=3)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.seg_lr, betas=(0.5, 0.999),)
        if self.cfg.sched:
            sched=LambdaLR(
                optimizer,
                lambda ep: max(1e-6, (1 - ep / self.cfg.num_epochs_seg) ** self.cfg.lr_scheduler_power))
            return [optimizer], [sched]
        return [optimizer], []

    def training_step(self, batch, batch_nb):
        rgb, label = (batch["source"], batch["source_segmentation"])

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()
            y_semseg_lbl = label.cuda()

        y_hat = self.net(rgb)
        # y_semseg_lbl=torch.clamp(y_semseg_lbl, min=0, max=2)

        loss_semseg = self.loss_semseg(y_hat, y_semseg_lbl)

        self.log_dict(
            {"loss_train/semseg": loss_semseg,},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": loss_semseg,
        }

    # def training_epoch_end(self, outputs):
    #     sampleimg=torch.rand((1,3,960,1280)).cuda()
    #     # self.logger.experiment[2].add_graph(self,sampleimg) TOFO
    #     pass

    def inference_step(self, batch):
        rgb = batch["source"]

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()

        y_hat = self.net(rgb)

        y_hat_semseg_lbl = y_hat.argmax(dim=1)

        return y_hat, y_hat_semseg_lbl

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl = self.inference_step(batch)
        y_semseg_lbl = batch["source_segmentation"]

        if torch.cuda.is_available() and self.cfg.gpu:
            y_semseg_lbl = y_semseg_lbl.cuda()

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)

        #  TODO
        self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)

        self.log_dict(
            {"loss_val/semseg": loss_val_semseg}, on_step=False, on_epoch=True,
        )

    def validation_epoch_end(self, outputs):
        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metric_semseg = metrics_semseg['MEAN IOU']

        scalar_logs = {
            "metrics_summary/semseg": metric_semseg,
        }
        scalar_logs.update(
            {
                f'metrics_task_semseg/{k.replace(" ", "_")}': v
                for k, v in metrics_semseg.items()
            }
        )

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        source_img, segmentation_img = (batch["source"], batch["source_segmentation"])

        y_seg = self.net(source_img)

        y_hat_semseg_lbl = y_seg.argmax(dim=1)
        self.metrics_semseg.update_batch(y_hat_semseg_lbl, segmentation_img)

        jaccard_index = self.iou_metric(y_seg, segmentation_img.int())

        logs = {
            "IOU Metric": jaccard_index,
        }

        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self,outputs):
        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metric_semseg = metrics_semseg['MEAN IOU']


        scalar_logs = {
            'metrics_test_summary/semseg': metric_semseg,
        }
        scalar_logs.update({f'metrics_test_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)
