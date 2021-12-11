import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

"""For semantic segmentation mask transforms
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
"""


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, segmentation):
        for t in self.transforms:
            image, segmentation = t(image, segmentation)
        return image, segmentation


class Resize:
    def __init__(self, img_size):
        self.img_size = img_size
        self.resize = T.Resize(img_size)
        self.resize_sem = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)

    def __call__(self, image, segmentation):
        image = self.resize(image)
        segmentation = self.resize_sem(segmentation)
        return image, segmentation


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, segmentation):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            segmentation = F.hflip(segmentation)
        return image, segmentation


class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, segmentation):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            segmentation = F.vflip(segmentation)
        return image, segmentation


class PILToTensor:
    def __call__(self, image, segmentation):
        image = F.to_tensor(image)
        segmentation = torch.as_tensor(np.array(segmentation), dtype=torch.int64)

        if len(segmentation.shape) > 2 and segmentation.shape[2] == 3:
            segmentation = segmentation[:, :, 0]

        return image, segmentation


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, segmentation):
        image = F.convert_image_dtype(image, self.dtype)
        return image, segmentation


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, segmentation):
        # this might create some values out of bounds
        image = F.normalize(image, mean=self.mean, std=self.std)
        # image=torch.clip(image,min=0,max=255)
        return image, segmentation


class RandomResizedCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, segmentation):
        # this might create some values out of bounds
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, segmentation


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomRotate:
    def __init__(self, degrees):
        self.degrees = degrees
        self.fill = 0

    def __call__(self, image, target):
        degrees = self.degrees
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item()
        )

        rotated_img = F.rotate(image, angle)
        rotated_target = F.rotate(target, angle)
        return rotated_img, rotated_target


class SegImageTransform:
    def __init__(self, img_size=64):
        self.transform = {
            "train": Compose(
                [
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    Resize((img_size*2, img_size*2)),
                    RandomRotate(degrees=[0, 45]),
                    RandomCrop(img_size),
                    # Resize((img_size, img_size)),
                    PILToTensor(),
                    Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
            "test": Compose(
                [
                    Resize((img_size, img_size)),
                    PILToTensor(),
                    Normalize(mean=[0.5], std=[0.5]),
                ]
            ),
        }

    def __call__(self, img, segmentation, phase="train"):
        img, segmentation = self.transform[phase](img, segmentation)

        return img, segmentation
