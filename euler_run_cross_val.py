from os import path
import uuid
import wandb

from datetime import datetime
from pytorch_lightning import Trainer
from datasets.generated import GeneratedDataModule
from datasets.mixed import MixedDataModule
from datasets.labeled import LabeledDataModule
from datasets.crossval import CrossValidationDataModule
from datasets.test import TestLabeledDataModule
from logger.gogoll_baseline_image import GogollBaselineImageLogger
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

from numpy import mean


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"{cfg.log_dir}/{run_name}/"

    data_dir = cfg.dataset_root

    # Config  -----------------------------------------------------------------
    batch_size = cfg.batch_size
    lr = {
        "G": 0.0002,
        "D": 0.0002,
        "seg_s": cfg.seg_lr,
        "seg_t": cfg.seg_lr,
    }
    epochs_seg = cfg.num_epochs_seg
    epochs_gogoll = cfg.num_epochs_gogoll
    reconstr_w = cfg.reconstruction_weight
    id_w = cfg.identity_weight
    seg_w = cfg.segmentation_weight
    if cfg.shared:
        wandb.init(
            reinit=True,
            name=run_name,
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
            entity="dlshared",
        )
    else:
        wandb.init(
            reinit=True,
            name=run_name,
            config=cfg,
            settings=wandb.Settings(start_method="fork"),
        )
    # Data Preprocessing  -----------------------------------------------------------------
    transform = SegImageTransform(img_size=cfg.image_size)

    # DataModule  -----------------------------------------------------------------
    dm = GogollDataModule(
        path.join(data_dir, 'source'), path.join(data_dir, 'easy', 'rgb'), transform, batch_size
    )  # used for training

    seg_dm = LabeledDataModule(
        path.join(data_dir, 'source'), transform, batch_size
    )

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
    seg_system = GogollSegSystem(seg_net_s, cfg=cfg)

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
        "seg_w": seg_w,
        "cfg": cfg,
    }
    main_system = GogollSystem(**gogoll_net_config)

    # Logger  --------------------------------------------------------------
    seg_wandb_logger = (
        WandbLogger(project=project_name, name=run_name, prefix="source_seg")
        if cfg.use_wandb
        else None
    )
    gogoll_wandb_logger = (
        WandbLogger(project=project_name, name=run_name, prefix="gogoll")
        if cfg.use_wandb
        else None
    )

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
    semseg_s_image_callback = GogollSemsegImageLogger(
        dm, network="net", log_key="Segmentation (Source)"
    )
    pipeline_image_callback = GogollPipelineImageLogger(dm, log_key="Pipeline")

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
        callbacks=[gogoll_checkpoint_callback, pipeline_image_callback,],
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
    )

    # Train
    if not cfg.seg_checkpoint_path:
        print("Fitting segmentation network...", run_name)
        seg_trainer.fit(seg_system, datamodule=seg_dm)
    else:
        print("Loading segmentation net from checkpoint...")
        seg_system = GogollSegSystem.load_from_checkpoint(
            cfg.seg_checkpoint_path, net=seg_net_s
        )

    if not cfg.gogoll_checkpoint_path:
        print("Fitting gogoll system...", run_name)
        trainer.fit(main_system, datamodule=dm)
    else:
        print("Loading gogol net from checkpoint...")
        main_system = GogollSystem.load_from_checkpoint(
            cfg.gogoll_checkpoint_path, **gogoll_net_config
        )

    generator = main_system.G_s2t

    # Image Generation & Saving  --------------------------------------------------------------
    if cfg.save_generated_images:
        dm_source = LabeledDataModule(
            path.join(data_dir, 'source'), transform, batch_size=batch_size, split=True
        )
        save_path = path.join(cfg.generated_dataset_save_root, run_name)
        # Generate fake target domain images and save them to a persistent folder (with the
        # same name as the current run)
        save_generated_dataset(
            generator,
            dm_source,
            save_path,
            logger=seg_wandb_logger,
            max_images=cfg.max_generated_images_saved,
        )

    # Train datamodules
    dm_source = LabeledDataModule(
        path.join(data_dir, 'source'), transform, batch_size=batch_size, split=True
    )
    dm_generated = GeneratedDataModule(generator, dm_source, batch_size=batch_size)
    
    # easy dataset with full dataset in test loader
    dm_easy_test = TestLabeledDataModule(
        path.join(data_dir, 'easy'), transform, batch_size=batch_size
    )

    n_splits = 5

    cv_source = CrossValidationDataModule(dm_generated, batch_size=batch_size, n_splits=n_splits)

    # train the final segmentation net that we use to evaluate if our augmented dataset helps
    # with training a segnet that is more robust to different domains/conditions
    n_cross_val_epochs = cfg.num_epochs_final

    evaluate_ours(
        cfg,
        project_name,
        cv_source,
        dm_easy_test,
        run_name,
        log_path,
        n_cross_val_epochs,
        n_splits,
    )

    wandb.finish()

# evaluate segementation on our generated data
def evaluate_ours(
    cfg,
    project_name,
    train_datamodule,
    test_datamodule,
    run_name,
    log_path,
    n_epochs,
    n_splits,
):
    # Cross Validation Run
    fold_metrics = {
        "iou": [],
        "soil": [],
        "weed": [],
        "crop": [],
    }

    for i in range(n_splits):
        # Cross Validation Run
        seg_net = UnetLight()
        seg_system = FinalSegSystem(seg_net, cfg=cfg)

        # Logger  --------------------------------------------------------------
        seg_wandb_logger = (
            WandbLogger(
                project=project_name,
                name=run_name,
            )
            if cfg.use_wandb
            else None
        )

        # Callbacks  --------------------------------------------------------------
        # save the model
        segmentation_checkpoint_callback = ModelCheckpoint(
            dirpath=path.join(log_path, f"segmentation_final_{i}"),
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor="loss",
            mode="min",
        )

        semseg_image_callback = GogollSemsegImageLogger(
            train_datamodule,
            network="net",
            log_key=f"Segmentation (Final) - Train",
        )

        baseline_image_callback = GogollBaselineImageLogger(
            test_datamodule,
            network="net",
            log_key=f"Segmentation (Final)",
        )

        cv_trainer = Trainer(
            max_epochs=n_epochs,
            gpus=1 if cfg.gpu else 0,
            reload_dataloaders_every_n_epochs=True,
            num_sanity_val_steps=0,
            logger=seg_wandb_logger,
            callbacks=[segmentation_checkpoint_callback, semseg_image_callback,baseline_image_callback],
        )

        cv_trainer.fit(seg_system, datamodule=train_datamodule)

        res = cv_trainer.test(seg_system, datamodule=test_datamodule)
        # Acess dict values of trainer after test and get metrics for average
        fold_metrics["iou"].append(res[0]["IOU Metric"])
        fold_metrics["soil"].append(res[0]["Test Metric Summary - soil"])
        fold_metrics["weed"].append(res[0]["Test Metric Summary - weed"])
        fold_metrics["crop"].append(res[0]["Test Metric Summary - crop"])

    # Acess dict values of trainer after test and get metrics for average
    wandb.run.summary[f"Crossvalidation IOU"] = mean(fold_metrics["iou"])
    wandb.run.summary[f"Crossvalidation IOU Soil"] = mean(fold_metrics["soil"])
    wandb.run.summary[f"Crossvalidation IOU Weed"] = mean(fold_metrics["weed"])
    wandb.run.summary[f"Crossvalidation IOU Crop"] = mean(fold_metrics["crop"])


if __name__ == "__main__":
    main()
