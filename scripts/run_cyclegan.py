import wandb
import uuid

from datetime import datetime
from pytorch_lightning import Trainer
from preprocessing.image_transform import ImageTransform
from datasets.agri import AgriDataModule
from systems.cycle_gan_system import CycleGANSystem
from models.generators import CycleGANGenerator
from models.discriminators import CycleGANDiscriminator
from logger.generated_image import GeneratedImageLogger
from utils.weight_initializer import init_weights
from configs.cyclegan_config import command_line_parser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"logs/{run_name}/"

    data_dir = cfg.dataset_root
    domain = cfg.domain

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    lr = {"G": 0.0002, "D": 0.0002}
    epoch = cfg.num_epochs
    reconstr_w = cfg.reconstruction_weight
    id_w = cfg.identity_weight

    # Data Preprocessing  -----------------------------------------------------------------
    transform = ImageTransform(img_size=cfg.image_size)

    # DataModule  -----------------------------------------------------------------
    dm = AgriDataModule(
        data_dir, transform, batch_size, domain=domain
    )  # used for training
    vs = AgriDataModule(
        data_dir, transform, batch_size, domain=domain
    )  # used for validation/progress visualization on wandb

    G_basestyle = CycleGANGenerator(filter=cfg.generator_filters)
    G_stylebase = CycleGANGenerator(filter=cfg.generator_filters)
    D_base = CycleGANDiscriminator(filter=cfg.discriminator_filters)
    D_style = CycleGANDiscriminator(filter=cfg.discriminator_filters)

    # Init Weight  --------------------------------------------------------------
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type="normal")

    # LightningModule  --------------------------------------------------------------
    model = CycleGANSystem(
        G_basestyle, G_stylebase, D_base, D_style, lr, transform, reconstr_w, id_w,
    )

    # Logger  --------------------------------------------------------------
    wandb_logger = (
        WandbLogger(project=project_name, name=run_name) if cfg.use_wandb else None
    )

    # Callbacks  --------------------------------------------------------------
    # save the model which had the best generator
    generator_checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        monitor="G_loss",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        filename="checkpoint/generator-{epoch:02d}-{G_loss:.4f}",
        verbose=False,
        mode="min",
    )

    # save the model which had the best discriminator
    discriminator_checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        monitor="D_loss",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        filename="checkpoint/discriminator-{epoch:02d}-{D_loss:.4f}",
        verbose=False,
        mode="min",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    generated_image_callback = GeneratedImageLogger(vs)

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    trainer = Trainer(
        max_epochs=epoch,
        gpus=1,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[
            generator_checkpoint_callback,
            discriminator_checkpoint_callback,
            generated_image_callback,
        ],
    )

    # Train
    print("Fitting", run_name)
    trainer.fit(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    main()
