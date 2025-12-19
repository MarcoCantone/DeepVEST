import torch
from matplotlib.scale import scale_factory
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, ConcatItemsd, EnsureChannelFirstd, AsDiscreted, ScaleIntensityd, \
    Orientationd, DeleteItemsd, RandSpatialCropSamples, Resize,RandSpatialCropSamplesd, LoadImage, RandAffined, \
    RandCropByPosNegLabeld, RandFlipd
from customTransform import separate_vessels_breast, SliceOrderingd
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
from config import load_config, create_object_from_dict
from persistent_homology import compute_betti
import time


cfg = load_config("/cache/marco/experiments/vessel_segmentation/8-unet/config.yaml")
data = torch.load("/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai_relative")
root = "/cache/marco/datasets/Duke-Breast-Cancer-MRI/"
for elem in data:
    for key in elem:
        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None
train_data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])


test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)
test_dataset = Dataset(test_data, test_transforms)
sample = test_dataset[5]

bin_img = sample["seg"]

for scale_factor in (np.arange(10)+1)/10:

    resize_dims = np.rint(np.array(bin_img[0].shape) * scale_factor)

    resized_bin = Resize(spatial_size=resize_dims, mode="nearest")(bin_img)[0]
    # TODO: try different interpolation method (less sparse output)

    plt.imshow(resized_bin.max(2).values, cmap="gray")
    plt.show()

    start = time.time()
    print(compute_betti(resized_bin))
    print(f'elapsed {time.time()-start} sec')

mip = bin_img[0].max(2).values
start = time.time()
print(compute_betti(mip))
print(f'elapsed {time.time() - start} sec')

