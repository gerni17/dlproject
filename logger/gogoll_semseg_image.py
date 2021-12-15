import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch
import wandb

from logger.semseg_image import prepare_semseg


class GogollSemsegImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(
        self,
        data_module,
        network="seg_s",
        log_key="Media/Segmentation (Source)",
        num_samples=4,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.network = network
        self.log_key = log_key

        if not data_module.has_prepared_data:
            data_module.prepare_data()

        if not data_module.has_setup_fit:
            data_module.setup()
            
        print(f'Loading {data_module.__class__.__name__}')
        dataloader = data_module.val_dataloader()

        val_samples = next(iter(dataloader))

        self.rgb_imgs = val_samples["source"]
        self.label_imgs = val_samples["source_segmentation"]
        self.label_imgs = prepare_semseg(self.label_imgs)

    def on_train_epoch_end(self, trainer, pl_module, *args):
        input_imgs = self.rgb_imgs.to(device=pl_module.device)
        labeled_imgs = self.label_imgs.to(device=pl_module.device)

        batch_size = input_imgs.shape[0]

        # get the segmentation network
        net = getattr(pl_module, self.network)

        # Get model prediction
        semseg = net(input_imgs)
        semseg = semseg.argmax(dim=1)
        semseg = prepare_semseg(semseg).to(device=pl_module.device)

        denorm_input_imgs = input_imgs * 0.5 + 0.5
        denorm_input_imgs = denorm_input_imgs * 255

        imgs = torch.cat([denorm_input_imgs, labeled_imgs, semseg], dim=0)

        joined_images_tensor = make_grid(imgs, nrow=batch_size, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1, 2, 0])

        try:
            # Log the images as wandb Image
            trainer.logger.experiment.log(
                {self.log_key: [wandb.Image(joined_images)]}, commit=False,
            )
        except BaseException as err:
            print(f"Error occured while uploading image to wandb. {err=}, {type(err)=}")
