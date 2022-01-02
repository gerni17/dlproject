from os import path
import uuid
import wandb

from datetime import datetime
from pytorch_lightning import Trainer
from datasets.gam import GamDataModule
from datasets.generated_gam import GeneratedGamDataModule
from datasets.labeled import LabeledDataModule
from datasets.crossval import CrossValidationDataModule
from datasets.test import TestLabeledDataModule
from logger.gam_pipeline_image import GamPipelineImageLogger
from logger.gogoll_baseline_image import GogollBaselineImageLogger
from logger.gogoll_pipeline_image import GogollPipelineImageLogger
from models.unet_light_semseg import UnetLight
from preprocessing.seg_transforms import SegImageTransform
from datasets.gogoll import GogollDataModule

from systems.final_seg_system import FinalSegSystem
from systems.gam_system import GamSystem
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
    }
    epochs_gogoll = cfg.num_epochs_gogoll
    reconstr_w = cfg.reconstruction_weight
    id_w = cfg.identity_weight
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
    dm = GamDataModule(
        path.join(data_dir, 'source'), path.join(data_dir, 'easy', 'rgb'), transform, batch_size
    )

    # Sub-Models  -----------------------------------------------------------------
    G_se2ta = CycleGANGenerator(filter=cfg.generator_filters, in_channels=3, out_channels=3)
    G_ta2se = CycleGANGenerator(filter=cfg.generator_filters, in_channels=3, out_channels=3)
    D_base = CycleGANDiscriminator(filter=cfg.discriminator_filters, in_channels=3)
    D_style = CycleGANDiscriminator(filter=cfg.discriminator_filters, in_channels=3)

    # Init Weight  --------------------------------------------------------------
    for net in [G_se2ta, G_ta2se, D_base, D_style]:
        init_weights(net, init_type="normal")

    # LightningModule  --------------------------------------------------------------

    gogoll_net_config = {
        "G_se2ta": G_se2ta,
        "G_ta2se": G_ta2se,
        "D_ta": D_base,
        "D_se": D_style,
        "lr": lr,
        "reconstr_w": reconstr_w
    }
    main_system = GamSystem(**gogoll_net_config)

    # Logger  --------------------------------------------------------------
    gam_wandb_logger = (
        WandbLogger(project=project_name, name=run_name, prefix="gam")
        if cfg.use_wandb
        else None
    )

    # Callbacks  --------------------------------------------------------------
    # save the model
    gam_checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(log_path, "gam"),
        save_last=False,
        save_top_k=1,
        verbose=True,
        monitor="loss_seg",
        mode="min",
    )

    # save the generated images (from the validation data) after every epoch to wandb
    pipeline_image_callback = GamPipelineImageLogger(dm, log_key="Pipeline")

    # Trainer  --------------------------------------------------------------
    print("Start training", run_name)
    print(f"Gpu {cfg.gpu}")

    trainer = Trainer(
        max_epochs=epochs_gogoll,
        gpus=1 if cfg.gpu else 0,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        logger=gam_wandb_logger,
        callbacks=[gam_checkpoint_callback, pipeline_image_callback,],
    )

    # Train
    if not cfg.gogoll_checkpoint_path:
        print("Fitting gam system...", run_name)
        trainer.fit(main_system, datamodule=dm)
    else:
        print("Loading gam net from checkpoint...")
        main_system = GamSystem.load_from_checkpoint(
            cfg.gogoll_checkpoint_path, **gogoll_net_config
        )

    generator = main_system.G_se2ta

    # Train datamodules
    dm_source = LabeledDataModule(
        path.join(data_dir, 'source'), transform, batch_size=batch_size, split=True
    )
    dm_generated = GeneratedGamDataModule(generator, dm_source, batch_size=batch_size)
    
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
        "GAM",
        "gam"
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
    experiment_name,
    model_save_name
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
                prefix=f"({experiment_name}) ",
            )
            if cfg.use_wandb
            else None
        )

        # Callbacks  --------------------------------------------------------------
        # save the model
        segmentation_checkpoint_callback = ModelCheckpoint(
            dirpath=path.join(log_path, f"segmentation_final_{model_save_name}_{i}"),
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor="loss",
            mode="min",
        )

        semseg_image_callback = GogollSemsegImageLogger(
            train_datamodule,
            network="net",
            log_key=f"Segmentation (Final - {experiment_name}) - Train",
        )

        baseline_image_callback = GogollBaselineImageLogger(
            test_datamodule,
            network="net",
            log_key=f"Segmentation (Final - {experiment_name})",
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
    wandb.run.summary[f"Crossvalidation IOU ({experiment_name})"] = mean(fold_metrics["iou"])
    wandb.run.summary[f"Crossvalidation IOU Soil ({experiment_name})"] = mean(fold_metrics["soil"])
    wandb.run.summary[f"Crossvalidation IOU Weed ({experiment_name})"] = mean(fold_metrics["weed"])
    wandb.run.summary[f"Crossvalidation IOU Crop ({experiment_name})"] = mean(fold_metrics["crop"])


if __name__ == "__main__":
    main()
