import os
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch import nn, optim
from utils.metrics import MetricsSemseg


class Semseg(pl.LightningModule):
    def __init__(self, cfg,net,lr):
        super(Semseg, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.net=net
        self.lr = lr

        self.loss_semseg = torch.nn.CrossEntropyLoss()
        names=["crop","weed", "soil"]
        self.metrics_semseg = MetricsSemseg(3, names)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
        )
        return [optimizer], []

    def training_step(self, batch, batch_nb):
        rgb, label = (batch["rgb"], batch["label"])

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()
            y_semseg_lbl = label.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat["semseg"]
        # y_semseg_lbl=torch.clamp(y_semseg_lbl, min=0, max=2)

        loss_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)

        self.log_dict({
                'loss_train/semseg': loss_semseg,
            }, on_step=True, on_epoch=True, prog_bar=True
        )

        return {
            'loss': loss_semseg,
        }


    # def training_epoch_end(self, outputs):
    #     sampleimg=torch.rand((1,3,960,1280)).cuda()
    #     # self.logger.experiment[2].add_graph(self,sampleimg) TOFO
    #     pass

    def inference_step(self, batch):
        rgb = batch["rgb"]

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat["semseg"]

        y_hat_semseg_lbl = y_hat_semseg.argmax(dim=1)

        return y_hat_semseg, y_hat_semseg_lbl

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl = self.inference_step(batch)
        y_semseg_lbl = batch["label"]

        if torch.cuda.is_available()  and self.cfg.gpu:
            y_semseg_lbl = y_semseg_lbl.cuda()

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
        loss_val_total = loss_val_semseg

        #  TODO
        # self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)

        self.log_dict({
                'loss_val/semseg': loss_val_semseg,
                'loss_val/total': loss_val_total,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        # metrics_semseg = self.metrics_semseg.get_metrics_summary()
        # self.metrics_semseg.reset()

        # metric_semseg = metrics_semseg['mean_iou']

        # metric_total = metric_semseg 

        # scalar_logs = {
        #     'metrics_summary/semseg': metric_semseg,
        #     'metrics_summary/total': metric_total,
        # }
        # scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})

        # self.log_dict(scalar_logs, on_step=False, on_epoch=True)
        pass

    def test_step(self, batch, batch_nb):
        pass
        # _, y_hat_semseg_lbl= self.inference_step(batch)
        # path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        # path_pred_semseg = os.path.join(path_pred, MOD_SEMSEG)
        # if batch_nb == 0:
        #     os.makedirs(path_pred_semseg)
        # split_test = SPLIT_TEST
        # for i in range(y_hat_semseg_lbl.shape[0]):
        #     sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
        #     path_file_semseg = os.path.join(path_pred_semseg, f'{sample_name}.png')
        #     pred_semseg = y_hat_semseg_lbl[i]
        #     self.datasets[split_test].save_semseg(
        #         path_file_semseg, pred_semseg, self.semseg_class_colors, self.semseg_ignore_label
        #     )

