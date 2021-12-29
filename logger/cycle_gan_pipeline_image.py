import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
import wandb

from logger.semseg_image import prepare_semseg


class CycleGanPipelineImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(self, data_module, log_key="Media/Pipeline"):
        super().__init__()
        self.log_key = log_key

        if not data_module.has_prepared_data:
            data_module.prepare_data()

        if not data_module.has_setup_fit:
            data_module.setup()
        
        dataloader = data_module.val_dataloader()
        val_samples = next(iter(dataloader))

        self.rgb_imgs = val_samples["source"]

    def on_train_epoch_end(self, trainer, pl_module, *args):
        input_imgs = self.rgb_imgs.to(device=pl_module.device)

        batch_size = input_imgs.shape[0]

        # get the segmentation network
        G_s2t = getattr(pl_module, "G_s2t")
        G_t2s = getattr(pl_module, "G_t2s")

        generated_target = G_s2t(input_imgs)
        cycled_input = G_t2s(G_s2t(input_imgs))

        plant_imgs = torch.cat([input_imgs, cycled_input, generated_target], dim=0)

        plant_imgs = plant_imgs * 0.5 + 0.5
        plant_imgs = plant_imgs * 255

        imgs = plant_imgs

        joined_images_tensor = make_grid(imgs, nrow=batch_size, padding=2)

        # Pipeline stacks the images like this:
        # - Source domain
        # - Generated source domain Reconstruction
        # - Generated target domain
        # - Ground truth segmentation
        # - Generated target domain segmentation
        # - Generated source domain reconstruction segmentation
        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1, 2, 0])

        try:
            # Log the images as wandb Image
            trainer.logger.experiment.log(
                {self.log_key: [wandb.Image(joined_images)]}, commit=False,
            )

        except BaseException as err:
            print(f"Error occured while uploading image to wandb. {err=}, {type(err)=}")
