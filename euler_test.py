from logging import log
from os import path
from torch.utils import data
import wandb
import uuid

from datetime import datetime
from pytorch_lightning import Trainer
from datasets.generated import GeneratedDataModule
from datasets.mixed import MixedDataModule
from datasets.source import SourceDataModule
from logger.gogoll_pipeline_image import GogollPipelineImageLogger
from models.lightweight_semseg import LightweightSemsegModel
from models.unet_light_semseg import UnetLight
from preprocessing.seg_transforms import SegImageTransform
from datasets.gogoll import GogollDataModule

from logger.generated_image import GeneratedImageLogger
from systems.final_seg_system import FinalSegSystem
from systems.gogoll_seg_system import GogollSegSystem
from utils.generate_targets_with_semantics import save_generated_dataset
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
    seg_s_lr = 0.0002
    epochs_seg = cfg.num_epochs_seg

    # Data Preprocessing  -----------------------------------------------------------------
    transform = SegImageTransform(img_size=cfg.image_size)

    # DataModule  -----------------------------------------------------------------
    dm = GogollDataModule(data_dir, cfg.domain, transform, batch_size)  # used for training
    vs = GogollDataModule(data_dir, cfg.domain, transform, batch_size)  # used for validation/progress visualization on wandb

    # Sub-Models  -----------------------------------------------------------------
    seg_net_s = UnetLight()
    seg_net_t = UnetLight()
    G_basestyle = CycleGANGenerator(filter=cfg.generator_filters)
    G_stylebase = CycleGANGenerator(filter=cfg.generator_filters)
    D_base = CycleGANDiscriminator(filter=cfg.discriminator_filters)
    D_style = CycleGANDiscriminator(filter=cfg.discriminator_filters)

    # Init Weight  --------------------------------------------------------------
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type="normal")

    # LightningModule  --------------------------------------------------------------
    seg_system = GogollSegSystem(seg_net_s, lr=seg_s_lr)

    # Logger  --------------------------------------------------------------
    seg_wandb_logger = WandbLogger(project=project_name, name=run_name, prefix="seg") if cfg.use_wandb else None
    wandb.login(key="969803cb62211763351a441ac5c9e96ce995f7eb") # for aleks euler

    # Callbacks  --------------------------------------------------------------
    # save the model
    segmentation_checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(log_path, "segmentation"),
        save_last=False,
        save_top_k=1,
        verbose=False,
        monitor="loss",
        mode="min",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    semseg_s_image_callback = GogollSemsegImageLogger(vs, network="net", log_key="Segmentation (Source)")

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    print(f"Gpu {cfg.gpu}")
    seg_trainer = Trainer(
        max_epochs=epochs_seg,
        gpus=1 if cfg.gpu else 0,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=seg_wandb_logger,
        callbacks=[
            segmentation_checkpoint_callback,
            semseg_s_image_callback,
            # semseg_t_image_callback,
        ],
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    # Train
    seg_trainer.fit(seg_system, datamodule=dm)



    wandb.finish()



if __name__ == "__main__":
    main()
