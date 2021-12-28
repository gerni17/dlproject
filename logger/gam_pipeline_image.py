import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
import wandb

from logger.semseg_image import prepare_semseg


class GamPipelineImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(self, data_module, log_key="Media/Pipeline", num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        self.log_key = log_key

        if not data_module.has_prepared_data:
            data_module.prepare_data()

        if not data_module.has_setup_fit:
            data_module.setup()
        
        dataloader = data_module.val_dataloader()
        val_samples = next(iter(dataloader))

        self.target_imgs = val_samples["target"]
        self.seg_imgs = val_samples["source_segmentation"]
        self.seg_imgs = self.seg_imgs.float()

    def on_train_epoch_end(self, trainer, pl_module, *args):
        target_imgs = self.target_imgs.to(device=pl_module.device)
        seg_imgs = self.seg_imgs.to(device=pl_module.device)

        batch_size = target_imgs.shape[0]

        # get the generators
        G_se2ta = getattr(pl_module, "G_se2ta")
        G_ta2se = getattr(pl_module, "G_ta2se")

        generated_target = G_se2ta(seg_imgs)
        cycled_segmentation = G_ta2se(G_se2ta(seg_imgs))

        seg_imgs_display = prepare_semseg(seg_imgs.argmax(dim=1))
        seg_imgs_display = seg_imgs_display.to(device=pl_module.device)

        cycled_segmentation_display = prepare_semseg(cycled_segmentation.argmax(dim=1))
        cycled_segmentation_display = cycled_segmentation_display.to(device=pl_module.device)

        plant_imgs = generated_target
        plant_imgs = plant_imgs * 0.5 + 0.5
        plant_imgs = plant_imgs * 255

        imgs = torch.cat([seg_imgs_display, cycled_segmentation_display, plant_imgs], dim=0)

        joined_images_tensor = make_grid(imgs, nrow=batch_size, padding=2)

        # Pipeline stacks the images like this:
        # - Segmentation domain
        # - Segmentation domain Reconstruction
        # - Generated target domain
        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1, 2, 0])

        try:
            # Log the images as wandb Image
            trainer.logger.experiment.log(
                {self.log_key: [wandb.Image(joined_images)]}, commit=False,
            )

        except BaseException as err:
            print(f"Error occured while uploading image to wandb. {err=}, {type(err)=}")
