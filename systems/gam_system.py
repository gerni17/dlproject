import warnings

from torch.optim.lr_scheduler import LambdaLR

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
from torch import nn, optim
import pytorch_lightning as pl


class GamSystem(pl.LightningModule):
    def __init__(
        self,
        G_se2ta,  # generator segmentation to target
        G_ta2se,  # generator target to segmentation
        D_ta,
        D_se,
        lr,
        reconstr_w=10,  # reconstruction weighting
    ):
        super(GamSystem, self).__init__()
        self.G_se2ta = G_se2ta
        self.G_ta2se = G_ta2se
        self.D_ta = D_ta
        self.D_se = D_se
        self.lr = lr
        self.reconstr_w = reconstr_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
        self.cse_loss = nn.CrossEntropyLoss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        # self.semseg_loss = torchmetrics.IoU(3)
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.G_se2ta_optimizer = optim.Adam(
            self.G_se2ta.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.G_ta2se_optimizer = optim.Adam(
            self.G_ta2se.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.D_ta_optimizer = optim.Adam(
            self.D_ta.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.D_se_optimizer = optim.Adam(
            self.D_se.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )

        return (
            [
                self.G_se2ta_optimizer,
                self.G_ta2se_optimizer,
                self.D_ta_optimizer,
                self.D_se_optimizer,
            ],
            [],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        segmentation_img, target_img = (
            batch["source_segmentation"],
            batch["target"],
        )

        segmentation_img_target = segmentation_img.long()
        segmentation_img = segmentation_img.float()
        segmentation_img_noise = torch.randn(segmentation_img.shape) * 0.5 + 1
        segmentation_img_noise = segmentation_img_noise.to(device=self.device)

        b = target_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        fake_se = self.G_ta2se(target_img)
        fake_ta = self.G_se2ta(segmentation_img * segmentation_img_noise)
        cycled_se = self.G_ta2se(fake_ta)
        cycled_ta = self.G_se2ta(fake_se)

        if optimizer_idx == 0 or optimizer_idx == 1 or optimizer_idx == 4:
            # Train Generator
            # Validity
            # MSELoss
            val_ta = self.generator_loss(self.D_ta(fake_ta), valid)
            val_se = self.generator_loss(self.D_se(fake_se), valid)
            val_loss = (val_ta + val_se) / 2

            # Reconstruction
            reconstr_se = self.cse_loss(cycled_se, segmentation_img_target.argmax(dim=1))
            reconstr_ta = self.mae(cycled_ta, target_img)
            reconstr_loss = (reconstr_se + reconstr_ta) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss


            logs = {
                "G_loss": G_loss,
                "val_ta": val_ta,
                "val_se": val_se,
                "val_loss": val_loss,
                "reconstr_ta": reconstr_ta,
                "reconstr_se": reconstr_se,
                "reconstr_loss": reconstr_loss,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return G_loss

        elif optimizer_idx == 2 or optimizer_idx == 3:
            # Train Discriminator
            # MSELoss
            D_ta_gen_loss = self.discriminator_loss(
                self.D_ta(fake_ta), fake
            )
            D_se_gen_loss = self.discriminator_loss(
                self.D_se(fake_se), fake
            )
            D_ta_valid_loss = self.discriminator_loss(
                self.D_ta(target_img), valid
            )
            D_se_valid_loss = self.discriminator_loss(
                self.D_se(segmentation_img), valid
            )

            D_gen_loss = (D_ta_gen_loss + D_se_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_ta_valid_loss + D_se_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            logs = {
                "D_loss": D_loss,
                "D_ta_gen_loss": D_ta_gen_loss,
                "D_se_gen_loss": D_se_gen_loss,
                "D_ta_valid_loss": D_ta_valid_loss,
                "D_se_valid_loss": D_se_valid_loss,
                "D_gen_loss": D_gen_loss,
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return D_loss

    def training_epoch_end(self, outputs):
        self.step += 1

        return None

    def generate(self, inputs):
        return self.G_se2ta(inputs)
