import os
import shutil
import uuid

import boto3
from datetime import datetime

import requests
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TestTubeLogger, TensorBoardLogger
import wandb

from mtl.experiments.experiment_semseg import ExperimentSemseg
from mtl.utils.rules import check_all_rules, pack_submission
from mtl.utils.config import command_line_parser


def main():
    cfg = command_line_parser()
    torch.cuda.empty_cache()

    # Remove previous logs and check file structure
    if os.path.isdir(cfg.log_dir):
        shutil.rmtree(cfg.log_dir)
    check_all_rules(cfg) # might need new rules

    model = ExperimentSemseg(cfg)

    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'semseg_{timestamp}_{cfg.name}_{str(uuid.uuid4())[:5]}'
    log_path = f"logs/{run_name}/"

    tube_logger = TestTubeLogger(
        save_dir=os.path.join(cfg.log_dir),
        name='tube',
        version=0,
    )

    wandb_logger = WandbLogger(
        name=run_name,
        project='Semseg',
        save_dir=os.path.join(cfg.log_dir))
    tb_logger = TensorBoardLogger(
        name="tb",
        version="",
        save_dir=log_path,
        log_graph=True,
    )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, 'checkpoints'),
        save_last=False,
        save_top_k=1,
        monitor='loss_train/semseg',
        mode='max',
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        save_last=False,
        save_top_k=1,
        verbose=True,
        monitor='loss_train/semseg',
        mode='max',
    )

    if torch.cuda.is_available() and cfg.gpu:
        run_cuda =True

    print("Start training", run_name)
    trainer = Trainer(
        logger=[wandb_logger, tube_logger, tb_logger], #
        callbacks=[checkpoint_local_callback, checkpoint_callback],
        gpus='-1' if run_cuda else None,
        resume_from_checkpoint=cfg.resume,
        max_epochs=cfg.num_epochs,
        distributed_backend=None,
        weights_summary=None,
        weights_save_path=None,
        num_sanity_val_steps=1,
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=20,
        # limit_val_batches=10,
        # limit_test_batches=10,
        # log_every_n_steps=10,
    )
    print("fit")
    if not cfg.prepare_submission:
        trainer.fit(model)

    dir_pred = os.path.join(cfg.log_dir, 'predictions')
    shutil.rmtree(dir_pred, ignore_errors=True)
    trainer.test(model)


if __name__ == '__main__':
    main()
