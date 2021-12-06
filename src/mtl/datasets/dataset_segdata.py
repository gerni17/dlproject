import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from collections import namedtuple
from mtl.utils.transforms import get_transforms
from pathlib import Path


from mtl.datasets.definitions import *


class SegDataset(torch.utils.data.Dataset):
    """
    This is a segdataset inspired by SynScapes [1] dataset with RGB, Semantic, and Depth modalities.
    It provides a total of 2048 datasamples.
    [1]: @article{Synscapes,
             author={Magnus Wrenninge and Jonas Unger},
             title={Synscapes: A Photorealistic Synthetic Dataset for Street Scene Parsing},
             url={http://arxiv.org/abs/1810.08705},
             year={2018},
             month={Oct}
         }
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    SegDataClass = namedtuple('SegDataClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        SegDataClass('soil', 0, 0, 'flat', 0, False, False, (0, 0, 0)),
        SegDataClass('crop', 1, 0, 'object', 0, False, False, (0, 200, 0)),
        SegDataClass('weed', 2, 0, 'object', 0, False, False, (200, 0, 0)),

    ]

    def __init__(self, dataset_root, split, integrity_check=False):
        assert split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST), f'Invalid split {split}'
        self.dataset_root = dataset_root
        self.split = split
        self.transforms = None
        if integrity_check:
            for i in tqdm(range(len(self))):
                self.get(i)

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        # load rgb
        rgb = Image.open(self.get_item_path(index, MOD_RGB))
        rgb.load()
        assert rgb.mode == 'RGB'

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
        }

        # load semseg
        path_semseg = self.get_item_path(index, MOD_SEMSEG)
        if os.path.isfile(path_semseg):
            semseg = self.load_semseg(path_semseg)
            assert semseg.size == rgb.size
            out[MOD_SEMSEG] = semseg
        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            out = self.transforms(out)
        return out

    def get_item_path(self, index, modality):
        return os.path.join(
            self.dataset_root, self.split, modality, f'{index}.{"png" if modality == MOD_RGB else "png"}'
        )

    def name_from_index(self, index):
        return f'{index}'

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return {
            SPLIT_TRAIN: 1600, #0-1600
            SPLIT_VALID: 447, #1601-2047
            SPLIT_TEST: 100, #2048-2147
        }[self.split]

    # TODO check that this is really the correct mean and std_dev
    @property
    def rgb_mean(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        # return [255 * 0.485, 255 * 0.456, 255 * 0.406] original imagenet values
        return [31.91585741, 27.6638663,  21.54151065]

    @property
    def rgb_stddev(self):
        # imagenet statistics used in pretrained networks - these are allowed to not match stats of this dataset
        # return [255 * 0.229, 255 * 0.224, 255 * 0.225] original imagenet values
        return [6.87016723, 6.68624823, 4.44775208]

    @staticmethod
    def load_semseg(path):
        semseg = Image.open(path)
        assert semseg.mode in ('I')
        return semseg

    @staticmethod
    def save_semseg(path, img, semseg_color_map, semseg_ignore_label=None, semseg_ignore_color=(0, 0, 0)):
        if torch.is_tensor(img):
            img = img.squeeze()
            assert img.dim() == 2 and img.dtype in (torch.int, torch.long)
            img = img.cpu().byte().numpy()
            img = Image.fromarray(img, mode='P')
        palette = [0 for _ in range(256 * 3)]
        for i, rgb in enumerate(semseg_color_map):
            for c in range(3):
                palette[3 * i + c] = rgb[c]
        if semseg_ignore_label is not None:
            for c in range(3):
                palette[3 * semseg_ignore_label + c] = semseg_ignore_color[c]
        img.putpalette(palette)
        img.save(path, optimize=True)

    @property
    def semseg_num_classes(self):
        return len(self.semseg_class_names)

    @property
    def semseg_ignore_label(self):
        return 255

    @property
    def semseg_class_colors(self):
        return [clsdesc.color for clsdesc in self.classes if not clsdesc.ignore_in_eval]

    @property
    def semseg_class_names(self):
        return [clsdesc.name for clsdesc in self.classes if not clsdesc.ignore_in_eval]


if __name__ == '__main__':
    print('Checking dataset integrity...')
    path = Path(__file__).parents[3]
    path = os.path.join(path, 'data/exp')

    ds_train = SegDataset(path, SPLIT_TRAIN, integrity_check=False)
    ds_valid = SegDataset(path, SPLIT_VALID, integrity_check=True)
    ds_test = SegDataset(path, SPLIT_TEST, integrity_check=True)
    print(ds_train.set_transforms(get_transforms(
        ds_train.get(0))))
    print('Dataset integrity check passed')
