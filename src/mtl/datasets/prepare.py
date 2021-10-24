import os
import sys
import numpy as np
import torch
from PIL import Image
# from torchvision.datasets import Cityscapes
from tqdm import tqdm
from collections import namedtuple
import torchvision
from mtl.datasets.dataset_segdata import SegDataset
from pathlib import Path

def main():
    """iterate over files and robustly relabel the labels with 0,1,2 instead of 0,10000,20000"""
    p = Path(__file__).parents[3]
    path_lab = os.path.join(p, 'data/exp/SegData/Seg_Data/Labels')

    files=[name for name in os.listdir(path_lab) if os.path.isfile(os.path.join(path_lab, name))]

    for ind,file in enumerate(files):
        print(file)
        file=os.path.join(path_lab, file)
        image = Image.open(file)
        if image.mode !='I':
            image.convert('I')
        data = np.asarray(image)
        a=torch.tensor(data)
        a[a>19999]=20000
        a[a==10000]=1
        a[a==20000]=2
        
        trans = torchvision.transforms.ToPILImage()
        c=trans(a)
        c.save(file)

if __name__ == '__main__':
    main()