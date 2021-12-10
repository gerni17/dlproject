from os import path
import uuid
import wandb

from datetime import datetime
from pytorch_lightning import Trainer
from datasets.mixed import MixedDataModule
from datasets.source import SourceDataModule
from logger.gogoll_baseline_image import GogollBaselineImageLogger
from models.unet_light_semseg import UnetLight
from preprocessing.seg_transforms import SegImageTransform

from systems.final_seg_system import FinalSegSystem
from systems.gogoll_seg_system import GogollSegSystem
from utils.generate_targets_with_semantics import save_generated_dataset
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

    # Train datamodules
    dm_source = SourceDataModule(
        path.join(data_dir, 'exp', 'train'), transform, batch_size=batch_size, split=True, max_imgs=200
    )
    dm_easy_train = SourceDataModule(
        path.join(data_dir, 'easy', 'train'), transform, batch_size=batch_size, split=True, max_imgs=200
    )
    dm_medium_train = SourceDataModule(
        path.join(data_dir, 'medium', 'train'), transform, batch_size=batch_size, split=True, max_imgs=200
    )

    dm_all_easy = MixedDataModule(dm_source, dm_easy_train, batch_size=batch_size)
    dm_all_medium = MixedDataModule(dm_medium_train, dm_easy_train, batch_size=batch_size)

    # Test datamodules
    easy_test = SourceDataModule(
        path.join(data_dir, 'easy', 'test'), transform, batch_size=batch_size, split=False, max_imgs=200
    )
    medium_test = SourceDataModule(
        path.join(data_dir, 'medium', 'test'), transform, batch_size=batch_size, split=False, max_imgs=200
    )

    baselines = [
        {
            "train": dm_source,
            "test": easy_test,
            "name": "Source <> Easy",
        },
        {
            "train": dm_source,
            "test": medium_test,
            "name": "Source <> Medium",
        },
        {
            "train": dm_all_easy,
            "test": easy_test,
            "name": "All (Easy) <> Easy",
        },
        {
            "train": dm_all_medium,
            "test": medium_test,
            "name": "All (Medium) <> Medium",
        },
    ]

    n_epochs = 16

    for baseline in baselines:
        evaluate_baseline(
            baseline['name'],
            cfg,
            baseline['train'],
            baseline['test'],
            project_name,
            run_name,
            log_path,
            n_epochs
        )

    wandb.finish()


def evaluate_baseline(
    baseline_name,
    cfg,
    train_datamodule,
    test_datamodule,
    project_name,
    run_name,
    log_path,
    n_epochs,
):
    # Cross Validation Run
    seg_lr = 0.0002
    seg_net = UnetLight()
    seg_system = FinalSegSystem(seg_net, lr=seg_lr)
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
        dirpath=path.join(log_path, f"segmentation_final_{safe_baseline_name}"),
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
        log_key=f"Segmentation (Final) - Baseline {baseline_name}",
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
    wandb.run.summary[f"MEAN IOU - {baseline_name}"] = res[0]["IOU Metric"]


if __name__ == "__main__":
    main()
