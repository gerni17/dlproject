import wandb
import uuid

from datetime import datetime
from pytorch_lightning import Trainer
from logger.gogoll_semseg_image import GogollSemsegImageLogger
from preprocessing.seg_transforms import SegImageTransform
from datasets.labeled import LabeledDataModule

from logger.generated_image import GeneratedImageLogger

# from utils.weight_initializer import init_weights
from configs.seg_config import command_line_parser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from systems.experiment_semseg import Semseg
from models.unet_light_semseg import UnetLight


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"logs/{run_name}/"

    data_dir = cfg.dataset_root

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    lr = 0.0002
    epoch = cfg.num_epochs

    # Data Preprocessing  -----------------------------------------------------------------
    transform = SegImageTransform(img_size=cfg.image_size)
    wandb.init(
        reinit=True,
        name=run_name,
        config=cfg,
        settings=wandb.Settings(start_method="fork"),
    )

    # DataModule  -----------------------------------------------------------------
    dm = LabeledDataModule(data_dir, transform, batch_size,split=True,max_imgs=20)  # used for training
    vs = LabeledDataModule(
        data_dir, transform, batch_size
    )  # used for validation/progress visualization on wandb

    net = UnetLight()

    # LightningModule  --------------------------------------------------------------
    model = Semseg(cfg, net, lr)
    # Logger  --------------------------------------------------------------
    wandb_logger = (
        WandbLogger(project=project_name, name=run_name) if cfg.use_wandb else None
    )

    # Callbacks  --------------------------------------------------------------
    # save the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        save_last=False,
        save_top_k=2,
        verbose=False,
        monitor="loss_train/semseg",
        mode="max",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    semseg_image_callback = GogollSemsegImageLogger(
        vs, network="net", log_key="Segmentation",
    )

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    trainer = Trainer(
        max_epochs=epoch,
        gpus=1,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=wandb_logger if cfg.use_wandb else None,
        callbacks=[checkpoint_callback, semseg_image_callback],
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    # Train
    print("Fitting", run_name)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    main()
