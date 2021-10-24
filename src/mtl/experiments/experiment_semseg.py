import os
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from mtl.datasets.definitions import *
from mtl.losses.loss_regression import LossRegression
from mtl.utils.metrics import MetricsSemseg
from mtl.utils.helpers import resolve_optimizer, resolve_dataset_class, resolve_model_class, resolve_lr_scheduler
from mtl.utils.transforms import get_transforms
from mtl.utils.visualization import compose


class ExperimentSemseg(pl.LightningModule):
    def __init__(self, cfg):
        super(ExperimentSemseg, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        dataset_class = resolve_dataset_class(cfg.dataset)
        self.datasets = {
            split: dataset_class(cfg.dataset_root, split, integrity_check=True)
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')

        self.semseg_num_classes = self.datasets[SPLIT_TRAIN].semseg_num_classes
        self.semseg_ignore_label = self.datasets[SPLIT_TRAIN].semseg_ignore_label
        self.semseg_class_names = self.datasets[SPLIT_TRAIN].semseg_class_names
        self.semseg_class_colors = self.datasets[SPLIT_TRAIN].semseg_class_colors
        self.rgb_mean = self.datasets[SPLIT_TRAIN].rgb_mean
        self.rgb_stddev = self.datasets[SPLIT_TRAIN].rgb_stddev

        outputs_descriptor = {
            MOD_SEMSEG: self.semseg_num_classes,

        }

        model_class = resolve_model_class(cfg.model_name)
        self.net = model_class(cfg, outputs_descriptor)

        self.datasets[SPLIT_TRAIN].set_transforms(get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            geom_scale_min=cfg.aug_geom_scale_min,
            geom_scale_max=cfg.aug_geom_scale_max,
            geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
            geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
            geom_reflect=cfg.aug_geom_reflect,
            crop_random=cfg.aug_input_crop_size,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
        ))
        self.transforms_val_test = get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            crop_for_passable=32,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
        )
        self.datasets[SPLIT_VALID].set_transforms(self.transforms_val_test)
        self.datasets[SPLIT_TEST].set_transforms(self.transforms_val_test)

        self.loss_semseg = torch.nn.CrossEntropyLoss(ignore_index=self.semseg_ignore_label)

        self.metrics_semseg = MetricsSemseg(self.semseg_num_classes, self.semseg_ignore_label, self.semseg_class_names)

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()
            y_semseg_lbl = y_semseg_lbl.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]
        # y_semseg_lbl=torch.clamp(y_semseg_lbl, min=0, max=2)

        if type(y_hat_semseg) is list:
            # deep supervision scenario: penalize all predicitons in the list and average losses
            loss_semseg = sum([self.loss_semseg(y_hat_semseg_i, y_semseg_lbl) for y_hat_semseg_i in y_hat_semseg])
            loss_semseg = loss_semseg / len(y_hat_semseg)
            y_hat_semseg = y_hat_semseg[-1]
        else:
            loss_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)

        loss_total = self.cfg.loss_weight_semseg * loss_semseg 

        self.log_dict({
                'loss_train/semseg': loss_semseg,
                'loss_train/total': loss_total,
            }, on_step=True, on_epoch=False, prog_bar=True
        )
        if self.can_visualize():
            self.visualize(batch, y_hat_semseg, batch[MOD_ID], 'imgs_train/batch_crops')

        return {
            'loss': loss_total,
        }

    def inference_step(self, batch):
        rgb = batch[MOD_RGB]

        if torch.cuda.is_available() and self.cfg.gpu:
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]

        if type(y_hat_semseg) is list:
            y_hat_semseg = y_hat_semseg[-1]

        y_hat_semseg_lbl = y_hat_semseg.argmax(dim=1)

        return y_hat_semseg, y_hat_semseg_lbl

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl = self.inference_step(batch)
        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)

        if torch.cuda.is_available()  and self.cfg.gpu:
            y_semseg_lbl = y_semseg_lbl.cuda()

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
        loss_val_total = loss_val_semseg

        #  removed because it produced error
        self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)

        self.log_dict({
                'loss_val/semseg': loss_val_semseg,
                'loss_val/total': loss_val_total,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        self.observer_step()
        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metric_semseg = metrics_semseg['mean_iou']

        metric_total = metric_semseg 
        metric_grader = (metric_semseg).clamp(min=0) # TODO check what it does/ is it really needed for sinle tasks

        scalar_logs = {
            'metrics_summary/semseg': metric_semseg,
            'metrics_summary/total': metric_total,
            'metrics_summary/grader': metric_grader,
            'trainer/LR': torch.tensor(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0]),
        }
        scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        _, y_hat_semseg_lbl= self.inference_step(batch)
        path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        path_pred_semseg = os.path.join(path_pred, MOD_SEMSEG)
        if batch_nb == 0:
            os.makedirs(path_pred_semseg)
        split_test = SPLIT_TEST
        for i in range(y_hat_semseg_lbl.shape[0]):
            sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
            path_file_semseg = os.path.join(path_pred_semseg, f'{sample_name}.png')
            pred_semseg = y_hat_semseg_lbl[i]
            self.datasets[split_test].save_semseg(
                path_file_semseg, pred_semseg, self.semseg_class_colors, self.semseg_ignore_label
            )

    def test_end(self, outputs):
        return {}

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

    def create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_VALID)

    def test_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_TEST)

    def visualize(self, batch, y_hat_semseg, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_semseg_lbl = y_hat_semseg.cpu().detach().argmax(dim=1)
        visualization_plan = [
            (MOD_RGB, batch[MOD_RGB], rgb_tags),
            (MOD_SEMSEG, batch[MOD_SEMSEG], 'GT SemSeg'),
            (MOD_SEMSEG, y_hat_semseg_lbl, 'Prediction SemSeg'),
        ]
        vis = compose(
            visualization_plan,
            self.cfg,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            semseg_color_map=self.semseg_class_colors,
            semseg_ignore_label=self.semseg_ignore_label,
        )
        # Use commit=False to not increment the step counter
        self.logger.experiment[0].log({
            tag: [wandb.Image(vis.cpu(), caption=tag)]
        }, commit=False)

        # Tensorboard logging
        self.logger.experiment[2].add_image(tag, vis, self.global_step)


    def can_visualize(self):
        return (not torch.cuda.is_available() or torch.cuda.current_device() == 0) and (
                self.global_step - self.cfg.num_steps_visualization_first) % \
                self.cfg.num_steps_visualization_interval == 0

    def observer_step(self):
        if torch.cuda.is_available() and torch.cuda.current_device() != 0:
            return
        vis_transforms = self.transforms_val_test
        list_samples = []
        for i in self.cfg.observe_train_ids:
            list_samples.append(self.datasets[SPLIT_TRAIN].get(i, override_transforms=vis_transforms))
        for i in self.cfg.observe_valid_ids:
            list_samples.append(self.datasets[SPLIT_VALID].get(i, override_transforms=vis_transforms))
        list_prefix = ('imgs_train/', ) * len(self.cfg.observe_train_ids) + ('imgs_val/', ) * len(self.cfg.observe_valid_ids)
        batch = default_collate(list_samples)
        rgb = batch[MOD_RGB]
        rgb_tags = [f'{prefix}{id}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
        with torch.no_grad():
            if torch.cuda.is_available()  and self.cfg.gpu:
                rgb = rgb.cuda()

            y_hat = self.net(rgb)
            y_hat_semseg = y_hat[MOD_SEMSEG]
            if type(y_hat_semseg) is list:
                y_hat_semseg = y_hat_semseg[-1]
        self.visualize(batch, y_hat_semseg, rgb_tags, 'imgs_val/observed_samples')
