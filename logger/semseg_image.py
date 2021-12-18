import numpy as np
from torchvision.utils import make_grid
from pytorch_lightning import Callback
import torch

import wandb


def prepare_semseg(img):
    assert (img.dim() == 3 or img.dim() == 4 and img.shape[1] == 1) and img.dtype in (
        torch.int,
        torch.long,
    ), f"Expecting 4D tensor with semseg classes, got {img.shape} and type {img.dtype}"
    if img.dim() == 4:
        img = img.squeeze(1)
    semseg_color_map = [(0, 0, 0), (0, 200, 0), (200, 0, 0)]
    colors = torch.tensor(semseg_color_map, dtype=torch.float32)
    assert colors.dim() == 2 and colors.shape[1] == 3
    if torch.max(colors) > 128:
        colors
    img = img.cpu().clone()  # N x H x W
    img = colors[img]  # N x H x W x 3
    img = img.permute(0, 3, 1, 2)
    return img


class SemsegImageLogger(Callback):
    """
    Callback which at the end of every training epoch will log some generated images to wandb.

    The images have the same input across all epochs, so you see the progression of how the generated images get better for a given input/source-image.
    """

    def __init__(self, data_module, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.test_dataloader()
        val_samples = next(iter(dataloader))

        self.rgb_imgs = val_samples["source"]
        self.label_imgs = val_samples["source_segmentation"]
        self.label_imgs = prepare_semseg(self.label_imgs)

    def on_validation_epoch_end(self, trainer, pl_module, *args):
        input_imgs = self.rgb_imgs.to(device=pl_module.device)
        labeled_imgs = self.label_imgs.to(device=pl_module.device)
        # Get model prediction
        semseg = pl_module.net(input_imgs)
        semseg = semseg.argmax(dim=1)
        semseg = prepare_semseg(semseg).to(device=pl_module.device)

        imgs = torch.cat([input_imgs, labeled_imgs, semseg], dim=0)

        joined_images_tensor = make_grid(imgs, nrow=8, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1, 2, 0])

        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {"End of epoch results": [wandb.Image(joined_images)]}, commit=False,
        )
