import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
from torch import nn, optim
import pytorch_lightning as pl


class GamSystem(pl.LightningModule):
    def __init__(
        self,
        G_se2so,  # generator source to target
        G_so2se,  # generator target to source
        D_so,
        D_se,
        lr,
        reconstr_w=10,  # reconstruction weighting
        id_w=2,  # identity weighting
    ):
        super(GamSystem, self).__init__()
        self.G_se2so = G_se2so
        self.G_so2se = G_so2se
        self.D_so = D_so
        self.D_se = D_se
        self.lr = lr
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
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
        self.G_se2so_optimizer = optim.Adam(
            self.G_se2so.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.G_so2se_optimizer = optim.Adam(
            self.G_so2se.parameters(), lr=self.lr["G"], betas=(0.5, 0.999)
        )
        self.D_so_optimizer = optim.Adam(
            self.D_so.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )
        self.D_se_optimizer = optim.Adam(
            self.D_se.parameters(), lr=self.lr["D"], betas=(0.5, 0.999)
        )

        return (
            [
                self.G_se2so_optimizer,
                self.G_so2se_optimizer,
                self.D_so_optimizer,
                self.D_se_optimizer,
            ],
            [],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        source_img, segmentation_img = (
            batch["source_segmentation"],
            batch["target"],
        )

        b = source_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        fake_se = self.G_so2se(source_img)
        fake_so = self.G_se2so(segmentation_img)
        cycled_se = self.G_so2se(fake_so)
        cycled_so = self.G_se2so(fake_se)

        if optimizer_idx == 0 or optimizer_idx == 1 or optimizer_idx == 4:
            # Train Generator
            # Validity
            # MSELoss
            val_so = self.generator_loss(self.D_so(fake_so), valid)
            val_se = self.generator_loss(self.D_se(fake_se), valid)
            val_loss = (val_so + val_se) / 2

            # Reconstruction
            reconstr_se = self.mae(cycled_se, segmentation_img)
            reconstr_so = self.mae(cycled_so, source_img)
            reconstr_loss = (reconstr_se + reconstr_so) / 2

            # Identity
            id_so = self.mae(self.G_so2se(segmentation_img), segmentation_img)
            id_se = self.mae(self.G_se2so(source_img), source_img)
            id_loss = (id_so + id_se) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss


            logs = {
                "G_loss": G_loss,
                "val_so": val_so,
                "val_se": val_se,
                "val_loss": val_loss,
                "reconstr_so": reconstr_so,
                "reconstr_se": reconstr_se,
                "reconstr_loss": reconstr_loss,
                "id_so": id_so,
                "id_se": id_se,
                "id_loss": id_loss
            }

            self.log_dict(
                logs, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

            return G_loss

        elif optimizer_idx == 2 or optimizer_idx == 3:
            # Train Discriminator
            # MSELoss
            D_so_gen_loss = self.discriminator_loss(
                self.D_so(fake_so), fake
            )
            D_se_gen_loss = self.discriminator_loss(
                self.D_se(fake_se), fake
            )
            D_so_valid_loss = self.discriminator_loss(
                self.D_so(source_img), valid
            )
            D_se_valid_loss = self.discriminator_loss(
                self.D_se(segmentation_img), valid
            )

            D_gen_loss = (D_so_gen_loss + D_se_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_so_valid_loss + D_se_valid_loss) / 3

            # Count up
            self.cnt_train_step += 1

            logs = {
                "D_loss": D_loss,
                "D_so_gen_loss": D_so_gen_loss,
                "D_se_gen_loss": D_se_gen_loss,
                "D_so_valid_loss": D_so_valid_loss,
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
        return self.G_se2so(inputs)
