import os, glob, random
from shutil import copyfile
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import time

from torchvision.utils import make_grid

def main():
    save_to_folder = "E:\\Downloads\\Split\\uavzurich_medium"

    rgb_folder = "D:\\mugee\\dlproject\\tutorial\\data\\uavzurich_medium\\rgb"
    segmentation_folder = "D:\\mugee\\dlproject\\tutorial\\data\\uavzurich_medium\\semseg"

    rgb_paths = glob.glob(os.path.join(rgb_folder, "*.png"))
    seg_paths = []

    for p in rgb_paths:
      s_p = os.path.join(segmentation_folder, os.path.basename(p))
      seg_paths.append(s_p)

    rgb_train, rgb_test, seg_train, seg_test = train_test_split(rgb_paths, seg_paths, test_size=0.3)

    zipped_train = list(zip(rgb_train, seg_train))
    zipped_test = list(zip(rgb_test, seg_test))

    rgb_directory = os.path.join(save_to_folder, "train", "rgb")
    if not os.path.exists(rgb_directory):
        os.makedirs(rgb_directory)
    
    semseg_directory = os.path.join(save_to_folder, "train", "semseg")
    if not os.path.exists(semseg_directory):
        os.makedirs(semseg_directory)

    rgb_directory = os.path.join(save_to_folder, "test", "rgb")
    if not os.path.exists(rgb_directory):
        os.makedirs(rgb_directory)
    
    semseg_directory = os.path.join(save_to_folder, "test", "semseg")
    if not os.path.exists(semseg_directory):
        os.makedirs(semseg_directory)

    for idx, (p, s_p) in enumerate(zipped_train):
      copyfile(p, os.path.join(save_to_folder, "train", "rgb", f"{idx}.png"))
      copyfile(p, os.path.join(save_to_folder, "train", "semseg", f"{idx}.png"))

    for idx, (p, s_p) in enumerate(zipped_test):
      copyfile(p, os.path.join(save_to_folder, "test", "rgb", f"{idx}.png"))
      copyfile(p, os.path.join(save_to_folder, "test", "semseg", f"{idx}.png"))

if __name__ == "__main__":
    main()
