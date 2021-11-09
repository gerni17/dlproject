
from logging import log
import wandb
import uuid

from datetime import datetime
from pytorch_lightning import Trainer
from preprocessing.seg_transforms import SegImageTransform
from datasets.gogoll import GogollDataModule

from logger.generated_image import GeneratedImageLogger
from utils.weight_initializer import init_weights
from configs.gogoll_config import command_line_parser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from systems.experiment_semseg import Semseg
from models.semseg_model import ModelDeepLabV3Plus
from models.discriminators import CycleGANDiscriminator
from models.generators import CycleGANGenerator
from logger.gogoll_semseg_image import GogollSemsegImageLogger
from systems.gogoll_system import GogollSystem

def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"logs/{run_name}/"

    data_dir = cfg.dataset_root

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    lr = {
        "G": 0.0002,
        "D": 0.0002,
        "seg_s": 0.0002,
        "seg_t": 0.0002,
    }
    epoch = cfg.num_epochs
    reconstr_w = cfg.reconstruction_weight
    id_w = cfg.identity_weight

    # Data Preprocessing  -----------------------------------------------------------------
    transform = SegImageTransform(img_size=cfg.image_size)

    # DataModule  -----------------------------------------------------------------
    dm = GogollDataModule(data_dir, cfg.domain, transform, batch_size)  # used for training
    vs = GogollDataModule(data_dir, cfg.domain, transform, batch_size)  # used for validation/progress visualization on wandb

    # Sub-Models  -----------------------------------------------------------------
    seg_net_s = ModelDeepLabV3Plus(3)
    seg_net_t = ModelDeepLabV3Plus(3)
    G_basestyle = CycleGANGenerator(filter=cfg.generator_filters)
    G_stylebase = CycleGANGenerator(filter=cfg.generator_filters)
    D_base = CycleGANDiscriminator(filter=cfg.discriminator_filters)
    D_style = CycleGANDiscriminator(filter=cfg.discriminator_filters)

    # Init Weight  --------------------------------------------------------------
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type="normal")

    # LightningModule  --------------------------------------------------------------
    model = GogollSystem(
        G_basestyle,
        G_stylebase,
        D_base,
        D_style,
        seg_net_s,
        seg_net_t,
        lr,
        reconstr_w,
        id_w,
    )

    # Logger  --------------------------------------------------------------
    wandb_logger = WandbLogger(project=project_name, name=run_name) if cfg.use_wandb else None

    # Callbacks  --------------------------------------------------------------
    # save the model 
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        save_last=False,
        save_top_k=1,
        verbose=False,
        monitor='loss_train/semseg',
        mode='max',
    )

    # save the generated images (from the validation data) after every epoch to wandb
    semseg_s_image_callback = GogollSemsegImageLogger(vs, network="seg_s", log_key="Segmentation (Source)")
    # semseg_t_image_callback = GogollSemsegImageLogger(vs, network="seg_t", log_key="Segmentation (Target)")

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    print(f'Gpu {cfg.gpu}')
    trainer = Trainer(
        max_epochs=epoch,
        gpus=1 if cfg.gpu else 0,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            semseg_s_image_callback,
            # semseg_t_image_callback,
        ],
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    # Train
    print("Fitting", run_name)
    trainer.fit(model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    main()
