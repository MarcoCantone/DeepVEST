import os

import torch
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    DeleteItemsd
)

import matplotlib.pyplot as plt

# create a list of dict for the monai Dataset class
data = torch.load("/data/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative")
root_dataset = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
for elem in data:
    for key in elem:
        if isinstance(elem[key], str):
            elem[key] = os.path.join(root_dataset, elem[key]) if elem[key] is not None else None

# define the transforms
transform = Compose([
    LoadImaged(keys=["post_2", "breast"]),
    EnsureChannelFirstd(keys=["post_2", "breast"]),
    Orientationd(keys=["post_2", "breast"], axcodes="LPS"),
    ScaleIntensityd(keys=["post_2"]),
    DeleteItemsd(keys=["pre", "post_1", "post_3", "post_4", "seg"]),
])

# create the dataset object
dataset = Dataset(data, transform)

# read the first sample
sample = dataset[0]

# unpack image an label
# tensors are [channel, H, W, D]
img = sample['post_2']
breast_label = sample['breast']

# print img and label shape
print(f"The image shape is {img.shape}, the label shape is {breast_label.shape}. Must be equal.")

# show the MIP projection
plt.imshow(img[0].max(2).values, cmap='gray')
plt.show()
plt.imshow(breast_label[0].max(2).values, cmap='gray')
plt.show()

# show the slice 40
plt.imshow(img[0, :, :, 40], cmap='gray')
plt.show()
plt.imshow(breast_label[0, :, :, 40], cmap='gray')
plt.show()
