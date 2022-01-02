from os import path
import uuid
from scipy.sparse import base
import wandb

from datetime import datetime
from pytorch_lightning import Trainer
from datasets.crossval import CrossValidationDataModule
from datasets.mixed import MixedDataModule
from datasets.labeled import LabeledDataModule
from datasets.test import TestLabeledDataModule
from logger.gogoll_baseline_image import GogollBaselineImageLogger
from models.unet_light_semseg import UnetLight
from preprocessing.seg_transforms import SegImageTransform

from systems.final_seg_system import FinalSegSystem
from utils.weight_initializer import init_weights
from configs.gogoll_config import command_line_parser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from logger.gogoll_semseg_image import GogollSemsegImageLogger

from numpy import mean


def main():
    cfg = command_line_parser()

    timestamp = datetime.now().strftime("%m%d-%H%M")

    project_name = cfg.project
    run_name = f"{cfg.name}_{timestamp}_{str(uuid.uuid4())[:2]}"
    log_path = f"{cfg.log_dir}/{run_name}/"

    data_dir = cfg.dataset_root

    transform = SegImageTransform(img_size=cfg.image_size)

    batch_size = 8

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

    # Train datamodules
    dm_source = LabeledDataModule(
        path.join(data_dir, 'source'), transform, batch_size=batch_size, split=True
    )
    
    # easy dataset with a train/val/test split
    dm_easy_split = LabeledDataModule(
        path.join(data_dir, 'easy'), transform, batch_size=batch_size
    )
    # easy dataset with full dataset in test loader
    dm_easy_test = TestLabeledDataModule(
        path.join(data_dir, 'easy'), transform, batch_size=batch_size
    )

    n_splits = 5

    cv_source = CrossValidationDataModule(dm_source, batch_size=batch_size, n_splits=n_splits)
    cv_easy_split = CrossValidationDataModule(dm_easy_split, batch_size=batch_size, n_splits=n_splits)

    baselines = [
        {
            "train": cv_source,
            "test": dm_easy_test,
            "name": "Source <> Easy",
        },
        {
            "train": cv_easy_split, # no test datamodule because we use the same datamodule as train for test
            "name": "Easy <> Easy",
        },
    ]

    n_epochs = cfg.num_epochs_final

    for baseline in baselines:
        evaluate_baseline(
            cfg,
            baseline,
            project_name,
            run_name,
            log_path,
            n_epochs,
            n_splits=n_splits
        )

    wandb.finish()


def evaluate_baseline(
    cfg,
    baseline,
    project_name,
    run_name,
    log_path,
    n_epochs,
    n_splits,
):
    baseline_name = baseline['name']
    train_datamodule = baseline['train']
    test_datamodule = baseline.get('test') or baseline['train']

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
        safe_baseline_name = baseline_name.replace(' ', '_').replace('(', '').replace(')', '').replace('<>', 'to').lower()

        # Logger  --------------------------------------------------------------
        seg_wandb_logger = (
            WandbLogger(
                project=project_name,
                name=run_name,
                prefix=f"{baseline_name} ",
            )
            if cfg.use_wandb
            else None
        )

        # Callbacks  --------------------------------------------------------------
        # save the model
        segmentation_checkpoint_callback = ModelCheckpoint(
            dirpath=path.join(log_path, f"segmentation_final_{safe_baseline_name}_{i}"),
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor="loss",
            mode="min",
        )

        semseg_image_callback = GogollSemsegImageLogger(
            train_datamodule,
            network="net",
            log_key=f"Segmentation (Final) - Train {baseline_name}",
        )

        baseline_image_callback = GogollBaselineImageLogger(
            test_datamodule,
            network="net",
            log_key=f"Segmentation (Final) - Baseline Test Data - {baseline_name}",
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
    wandb.run.summary[f"Crossvalidation IOU - {baseline_name}"] = mean(fold_metrics["iou"])
    wandb.run.summary[f"Crossvalidation IOU Soil - {baseline_name}"] = mean(fold_metrics["soil"])
    wandb.run.summary[f"Crossvalidation IOU Weed - {baseline_name}"] = mean(fold_metrics["weed"])
    wandb.run.summary[f"Crossvalidation IOU Crop - {baseline_name}"] = mean(fold_metrics["crop"])


if __name__ == "__main__":
    main()
