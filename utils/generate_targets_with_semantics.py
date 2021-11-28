from datasets.sources import SourceDataModule
from torchvision.utils import save_image
from os import path

def save_generated_image(save_path, idx, img, segmentation):
    rgb_path = path.join(save_path, "rgb", f"{idx}.png")
    seg_path = path.join(save_path, "semseg", f"{idx}.png")

    save_image(img, rgb_path, "png")
    save_image(segmentation, seg_path, "png")


def undo_transform(image):
    return (image * 0.5 + 0.5) * 255


def generate_targets_with_semantics(system, data_dir, transform, save_path, max_images=10):
    print('Generating images...')

    dm = SourceDataModule(data_dir, transform, batch_size=1)
    idx = 0

    for sample in dm:
        source = sample["source"]
        segmentation = sample["source_segmentation"]

        target = system.generate(source)

        save_generated_image(save_path, idx, undo_transform(target), segmentation)

        idx = idx + 1

        if idx >= max_images:
            break
    
    print('Images generated successfully')
