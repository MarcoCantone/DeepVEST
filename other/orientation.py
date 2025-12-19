import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
from monai.data import Dataset
from monai.networks.nets.unet import Unet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, DeleteItemsd, ConcatItemsd, AsDiscrete
from monai.inferers import sliding_window_inference
from customTransform import SliceOrderingd, separate_vessels_breast
from config import load_config, create_object_from_dict

# bbs = pd.read_csv("/cache/marco/datasets/Duke-Breast-Cancer-MRI/Annotation_Boxes.csv")

data = torch.load(r"C:\Users\marco\Desktop\MRI\duke\data_monai_relative", weights_only=False)
root = r"C:\Users\marco\Desktop\MRI\duke"
for elem in data:
    for key in elem:
        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
data = test_data

ds = Dataset(data)

t1 = Compose([LoadImaged(keys=["post_2", "seg"]), EnsureChannelFirstd(["post_2", "seg"]), Orientationd(keys=["post_2", "seg"], axcodes="LPS")])
t2 = Compose([LoadImaged(keys=["post_2", "seg"]), EnsureChannelFirstd(["post_2", "seg"]), SliceOrderingd(img_key="post_2"), Orientationd(keys=["post_2", "seg"], axcodes="LPS")])
t4 = Compose([LoadImaged(keys=["post_2", "seg"]), EnsureChannelFirstd(["post_2", "seg"])])

cfg_path = r"C:\Users\marco\Desktop\config_tmp.yaml"
cfg = load_config(cfg_path)
test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)
t3 = test_transforms

for i in [0, 8]:
    n = 8
    sample = t3(ds[i])
    post2 = sample["img"][1]
    seg = sample["seg"][0]
    # post2 = sample["post_2"][0]
    # seg = sample["seg"][0]
    plt.figure(figsize=(4*n, 8))
    plt.suptitle("experiment")
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(post2[:, :, post2.shape[-1] // n * i], cmap="gray")
        plt.subplot(2, n, n+i+1)
        plt.imshow(seg[:, :, post2.shape[-1] // n * i], cmap="gray")
    plt.show()

