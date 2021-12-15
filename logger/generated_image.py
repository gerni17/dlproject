import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import wandb
import torch
from pytorch_lightning import Callback


class GeneratedImageLogger(Callback):
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

        self.source_imgs = val_samples["source"]

    def on_train_epoch_end(self, trainer, pl_module, *args):
        input_imgs = self.source_imgs.to(device=pl_module.device)
        # Get model prediction
        gen_imgs = pl_module.generate(input_imgs)
        gen_img = torch.cat([input_imgs, gen_imgs], dim=0)

        # Reverse Normalization
        gen_img = gen_img * 0.5 + 0.5
        gen_img = gen_img * 255

        joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

        joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
        joined_images = np.transpose(joined_images, [1, 2, 0])

        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {"End of epoch results": [wandb.Image(joined_images)]}, commit=False,
        )
