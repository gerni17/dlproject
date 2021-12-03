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
from datasets.mixedCV import MixedCrossValDataModule
from logger.gogoll_pipeline_image import GogollPipelineImageLogger
from models.unet_light_semseg import UnetLight
from preprocessing.seg_transforms import SegImageTransform
from datasets.gogoll import GogollDataModule

from systems.final_seg_system import FinalSegSystem
from systems.gogoll_seg_system import GogollSegSystem
from utils.generate_targets_with_semantics import save_generated_dataset
from utils.weight_initializer import init_weights
from configs.gogoll_config import command_line_parser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
    seg_s_lr = 0.0002
    epochs_seg = cfg.num_epochs_seg
    epochs_gogoll = cfg.num_epochs_gogoll
    reconstr_w = cfg.reconstruction_weight
    id_w = cfg.identity_weight
    seg_w = cfg.segmentation_weight

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

    gogoll_net_config = {
        "G_s2t": G_basestyle,
        "G_t2s": G_stylebase,
        "D_source": D_base,
        "D_target": D_style,
        "seg_s": seg_net_s,
        "seg_t": seg_net_t,
        "lr": lr,
        "reconstr_w": reconstr_w,
        "id_w": id_w,
        "seg_w": seg_w
    }
    main_system = GogollSystem(**gogoll_net_config)

    # Logger  --------------------------------------------------------------
    seg_wandb_logger = WandbLogger(project=project_name, name=run_name, prefix="seg") if cfg.use_wandb else None
    gogoll_wandb_logger = WandbLogger(project=project_name, name=run_name, prefix="gogoll") if cfg.use_wandb else None
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
    
    gogoll_checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(log_path, "gogoll"),
        save_last=False,
        save_top_k=1,
        verbose=True,
        monitor="loss_seg",
        mode="min",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    semseg_s_image_callback = GogollSemsegImageLogger(vs, network="net", log_key="Segmentation (Source)")
    pipeline_image_callback = GogollPipelineImageLogger(vs, log_key="Pipeline")

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

    trainer = Trainer(
        max_epochs=epochs_gogoll,
        gpus=1 if cfg.gpu else 0,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=gogoll_wandb_logger,
        callbacks=[
            gogoll_checkpoint_callback,
            pipeline_image_callback,
        ],
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    # Train
    if not cfg.seg_checkpoint_path:
        print("Fitting segmentation network...", run_name)
        seg_trainer.fit(seg_system, datamodule=dm)
    else:
        print(f"Loading segmentation net from checkpoint...")
        seg_system = GogollSegSystem.load_from_checkpoint(cfg.seg_checkpoint_path, net=seg_net_s)

    if not cfg.gogoll_checkpoint_path:
        print("Fitting gogoll system...", run_name)
        trainer.fit(main_system, datamodule=dm)
    else:
        print(f"Loading gogol net from checkpoint...")
        main_system = GogollSystem.load_from_checkpoint(cfg.gogoll_checkpoint_path, **gogoll_net_config)

    
    # Image Generation & Saving  --------------------------------------------------------------
    if cfg.save_generated_images:
        save_path = path.join(cfg.generated_dataset_save_root, run_name)
        # Generate fake target domain images and save them to a persistent folder (with the
        # same name as the current run)
        save_generated_dataset(main_system, data_dir, transform, save_path, logger=seg_wandb_logger, max_images=cfg.max_generated_images_saved)

    # Source domain datamodule
    source_dm = SourceDataModule(data_dir, transform, batch_size=1, max_imgs=200)
    # Generated images datamodule
    generated_dm = GeneratedDataModule(main_system.G_s2t, data_dir, transform, batch_size=1, max_imgs=200)
    
    # Mix both datamodules and do Cross Val
    n_splits = 5
    mixed_cv_dm = MixedCrossValDataModule(
        source_dm,
        generated_dm,
        batch_size=batch_size,
        n_splits=n_splits
    )
    log_cv_dm = MixedCrossValDataModule(
        source_dm,
        generated_dm,
        batch_size=batch_size,
        n_splits=n_splits
    )

    # train the final segmentation net that we use to evaluate if our augmented dataset helps
    # with training a segnet that is more robust to different domains/conditions
    n_cross_val_epochs = 10
    cross_val_final_segnet(cfg, mixed_cv_dm, log_cv_dm, project_name, run_name, log_path, n_cross_val_epochs, n_splits)

    wandb.finish()

def cross_val_final_segnet(cfg, datamodule, log_datamodule, project_name, run_name, log_path, n_epochs, n_splits):
    # Cross Validation Run
    fold_metrics = []
    for i in range(n_splits):
        datamodule.set_active_split(i)
        seg_net = UnetLight()
        seg_system = FinalSegSystem(seg_net, lr=seg_lr)

        seg_lr = 0.0002

        # Logger  --------------------------------------------------------------
        seg_wandb_logger = WandbLogger(project=project_name, name=run_name, prefix="seg_final_cv_fold{}".format(i)) if cfg.use_wandb else None

        # Callbacks  --------------------------------------------------------------
        # save the model
        segmentation_checkpoint_callback = ModelCheckpoint(
            dirpath=path.join(log_path, "segmentation_final_cv_fold{}".format(i)),
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor="loss",
            mode="min",
        )
        semseg_image_callback = GogollSemsegImageLogger(log_datamodule, network="net", log_key="Segmentation (Final) Fold {}".format(i))

    
        cv_trainer = Trainer(
            max_epochs=n_epochs,
            gpus=1 if cfg.gpu else 0,
            reload_dataloaders_every_n_epochs=True,
            num_sanity_val_steps=0,
            logger=seg_wandb_logger,
            callbacks=[
                segmentation_checkpoint_callback,
                semseg_image_callback,
            ],
        )

        print('------------final segmentation network fold no---------{}------------'.format(i))
        cv_trainer.fit(seg_system, datamodule=datamodule)
        print('------------testing fold no---------{}------------'.format(i))
        #cv_trainer.test(seg_system, datamodule=cv_dm)
        #Acess dict values of trainer after test and get metrics for average
        #fold_metrics.append(...)


if __name__ == "__main__":
    main()
