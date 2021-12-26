import warnings

from torch.optim.lr_scheduler import LambdaLR

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


class GogollSystem(pl.LightningModule):
    def __init__(
        self,
        G_s2t,  # generator source to target
        G_t2s,  # generator target to source
        D_source,
        D_target,
        seg_s,  # segmentation source
        seg_t,  # segmentation target
        lr,
        reconstr_w=10,  # reconstruction weighting
        id_w=2,  # identity weighting
        seg_w=1,
        cfg=None
    ):
        super(GogollSystem, self).__init__()
        self.G_s2t = G_s2t
        self.G_t2s = G_t2s
        self.D_source = D_source
        self.D_target = D_target
        self.seg_s = seg_s
        self.seg_t = seg_t
        self.lr = lr
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.seg_w = seg_w
        self.cnt_train_step = 0
        self.step = 0
        self.cfg = cfg
        self.initial_epoch = 9999999

        self.seg_t.load_state_dict(self.seg_s.state_dict())

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.semseg_loss = nn.CrossEntropyLoss()
        # self.semseg_loss = torchmetrics.IoU(3)
        self.losses = []
        self.Seg_mean_losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        # self.global_optimizer = optim.Adam(
        #     chain(
        #         self.G_s2t.parameters(),
        #         self.G_t2s.parameters(),
        #         self.D_source.parameters(),
        #         self.D_target.parameters(),
        #         self.seg_t.parameters(),
        #     ),
        #     lr=self.lr["G"],
        #     betas=(0.5, 0.999),
        # )
        self.g_s2t_optimizer = optim.Adam(
            self.G_s2t.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.g_t2s_optimizer = optim.Adam(
            self.G_t2s.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.d_source_optimizer = optim.Adam(
            self.D_source.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.d_target_optimizer = optim.Adam(
            self.D_target.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.seg_s_optimizer = optim.Adam(
            self.seg_s.parameters(), lr=self.lr["seg_s"], betas=(0.5, 0.999)
        )
        self.seg_t_optimizer = optim.Adam(
            self.seg_t.parameters(), lr=self.lr["seg_t"]/self.cfg.lr_ratio, betas=(0.5, 0.999)
        )

        # sched = LambdaLR(self.seg_t_optimizer, lambda ep: max(1e-6, (1 - (ep - self.initial_epoch) / self.cfg.num_epochs_gogoll) ** self.cfg.lr_scheduler_power))

        # self.g_optimizer = optim.Adam(chain(self.G_s2t.parameters(), self.G_t2s.parameters()), lr=self.lr["G"], betas=(0.5, 0.999))
        # self.d_optimizer = optim.Adam(chain(self.D_source.parameters(), self.D_target.parameters()), lr=self.lr["D"], betas=(0.5, 0.999))
        # self.seg_optimizer = optim.Adam(chain(self.seg_s.parameters(), self.seg_t.parameters()), lr=self.lr["D"], betas=(0.5, 0.999))

        return (
            [
                # self.global_optimizer,
                self.g_s2t_optimizer,
                self.g_t2s_optimizer,
                self.d_source_optimizer,
                self.d_target_optimizer,
                self.seg_t_optimizer,
                # self.seg_t_optimizer,
                # self.g_optimizer,
                # self.d_optimizer,
                # self.seg_optimizer,
            ],
            [],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.current_epoch > self.initial_epoch:
            self.initial_epoch = self.current_epoch

        source_img, segmentation_img, target_img = (
            batch["source"],
            batch["source_segmentation"],
            batch["target"],
        )

        b = source_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        fake_source = self.G_t2s(target_img)
        fake_target = self.G_s2t(source_img)
        cycled_source = self.G_t2s(fake_target)
        cycled_target = self.G_s2t(fake_source)

        if optimizer_idx == 0 or optimizer_idx == 1 or optimizer_idx == 4:
            # Train Generator
            # Validity
            # MSELoss
            val_source = self.generator_loss(self.D_source(fake_source), valid)
            val_target = self.generator_loss(self.D_target(fake_target), valid)
            val_loss = (val_source + val_target) / 2

            # Reconstruction
            reconstr_source = self.mae(cycled_source, source_img)
            reconstr_target = self.mae(cycled_target, target_img)
            reconstr_loss = (reconstr_source + reconstr_target) / 2

            # Identity
            id_source = self.mae(self.G_t2s(source_img), source_img)
            id_target = self.mae(self.G_s2t(target_img), target_img)
            id_loss = (id_source + id_target) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            # Segmentation
            y_seg_s = self.seg_s(cycled_source)
            y_seg_t = self.seg_t(fake_target)

            y_seg_t_s = self.seg_s(fake_source)
            y_seg_t_t = self.seg_t(target_img)
            y_seg_t_t_c = self.seg_t(cycled_target)

            loss_seg_a = self.semseg_loss(y_seg_s, segmentation_img)
            loss_seg_b = self.semseg_loss(y_seg_t, segmentation_img)
            loss_seg_c = self.semseg_loss(y_seg_t_t, y_seg_t_s.argmax(dim=1).long())
            loss_seg_d = self.semseg_loss(y_seg_t_t_c, y_seg_t_s.argmax(dim=1).long())

            Seg_loss = (loss_seg_a + loss_seg_b + loss_seg_c + loss_seg_d) / 4

            logs = {
                "G_loss": G_loss,
                "val_source": val_source,
                "val_target": val_target,
                "val_loss": val_loss,
                "reconstr_source": reconstr_source,
                "reconstr_target": reconstr_target,
                "reconstr_loss": reconstr_target,
                "id_source": reconstr_target,
                "id_target": id_target,
                "id_loss": id_loss,
                "loss_seg": Seg_loss,
                "loss_seg_a": loss_seg_a,
                "loss_seg_b": loss_seg_b,
                "loss_seg_c": loss_seg_c,
                "loss_seg_d": loss_seg_d,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return 4 * G_loss + Seg_loss * self.seg_w

        elif optimizer_idx == 2 or optimizer_idx == 3:
            # Train Discriminator
            # MSELoss
            D_source_gen_loss = self.discriminator_loss(
                self.D_source(fake_source), fake
            )
            D_target_gen_loss = self.discriminator_loss(
                self.D_target(fake_target), fake
            )
            D_source_valid_loss = self.discriminator_loss(
                self.D_source(source_img), valid
            )
            D_target_valid_loss = self.discriminator_loss(
                self.D_target(target_img), valid
            )

            D_gen_loss = (D_source_gen_loss + D_target_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_source_valid_loss + D_target_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            logs = {
                "D_loss": D_loss,
                "D_source_gen_loss": D_source_gen_loss,
                "D_target_gen_loss": D_target_gen_loss,
                "D_source_valid_loss": D_source_valid_loss,
                "D_target_valid_loss": D_target_valid_loss,
                "D_gen_loss": D_gen_loss,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return D_loss

    def training_epoch_end(self, outputs):
        self.step += 1

        # # all = range(6)
        # # segmentations = [0, 1]
        # # generators = [2, 3]
        # # discriminators = [4, 5]

        # # avg_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 4 for i in all])
        # # Seg_mean_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2 for i in segmentations])
        # # G_mean_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2 for i in generators])
        # # D_mean_loss = sum([torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2 for i in discriminators])
        # # validity = sum([torch.stack([x["validity"] for x in outputs[i]]).mean().item() / 2 for i in generators])
        # # reconstr = sum([torch.stack([x["reconstr"] for x in outputs[i]]).mean().item() / 2 for i in generators])
        # # identity = sum([torch.stack([x["identity"] for x in outputs[i]]).mean().item() / 2 for i in generators])

        # # self.losses.append(avg_loss)
        # # self.Seg_mean_losses.append(Seg_mean_loss)
        # # self.G_mean_losses.append(G_mean_loss)
        # # self.D_mean_losses.append(D_mean_loss)
        # # self.validity.append(validity)
        # # self.reconstr.append(reconstr)
        # # self.identity.append(identity)

        # self.losses.append(outputs[0]["loss"])

        return None

    def generate(self, inputs):
        return self.G_s2t(inputs)
