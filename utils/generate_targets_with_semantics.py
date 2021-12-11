import torch
from datasets.generated import GeneratedDataModule
from torchvision.utils import save_image, make_grid, Image
from os import path, makedirs
import wandb
import numpy as np


def save_segmentation(save_path, idx, segmentation):
    seg_path = path.join(save_path, "semseg", f"{idx}.png")

    directory_seg = path.dirname(seg_path)

    if not path.exists(directory_seg):
        makedirs(directory_seg)

    grid = make_grid(segmentation)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(seg_path, format="png")


def save_generated_image(save_path, idx, img):
    rgb_path = path.join(save_path, "rgb", f"{idx}.png")

    directory_rgb = path.dirname(rgb_path)

    if not path.exists(directory_rgb):
        makedirs(directory_rgb)

    save_image(img, rgb_path, "png")


def undo_transform(image):
    return image * 0.5 + 0.5


def save_generated_dataset(
    generator, source_dm, save_path, max_images=10, logger=None
):
    print("Generating images...")

    dm = GeneratedDataModule(generator, source_dm, batch_size=1)
    dm.prepare_data()
    dm.setup()
    ds = dm.train_dataloader().dataset
    idx = 0

    for sample in ds:
        source = sample["source"]
        segmentation = sample["source_segmentation"]
        shape = source.shape
        gen = undo_transform(torch.reshape(source, (shape[1], shape[2], shape[3])))

        save_generated_image(save_path, idx, gen)
        save_segmentation(save_path, idx, segmentation.float())

        idx = idx + 1

        if idx >= max_images:
            break
    print("Images generated successfully")

