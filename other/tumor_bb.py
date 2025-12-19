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

bbs = pd.read_csv("/cache/marco/datasets/Duke-Breast-Cancer-MRI/Annotation_Boxes.csv")

data = torch.load("/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai_relative", weights_only=False)
root = "/cache/marco/datasets/Duke-Breast-Cancer-MRI/"
for elem in data:
    for key in elem:
        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
data = test_data

ds = Dataset(data)

trans = Compose([
    LoadImaged(keys=["post_2", "seg"]),
    EnsureChannelFirstd(keys=["post_2", "seg"]),
    SliceOrderingd(img_key="post_2"),
    separate_vessels_breast(seg_key='seg', channel=True, compute_backgroud=False, dense=False, adjust_affine="post_2"),
    Orientationd(keys=["post_2", "vessels"], axcodes="LPS"),
    DeleteItemsd(keys=["pre", "post_1", "post_3", "post_4", "breast", "seg"]),
])

trans1 = Compose([
    LoadImaged(keys=["pre", "post_2", "seg"]),
    EnsureChannelFirstd(keys=["pre", "post_2", "seg"]),
    ConcatItemsd(keys=["pre", "post_2"], name="img", dim=0),
    SliceOrderingd(img_key="post_2"),
    separate_vessels_breast(seg_key='seg', channel=True, compute_backgroud=False, dense=False, adjust_affine="post_2"),
    Orientationd(keys=["img", "post_2", "vessels"], axcodes="LPS"),
    DeleteItemsd(keys=["pre", "post_1", "post_3", "post_4", "breast", "seg"]),
])

model = Unet(in_channels=2, out_channels=2, norm="BATCH", spatial_dims=3, channels=[32, 64, 128, 256], strides=[2, 2, 2])
model.load_state_dict(torch.load("/cache/marco/experiments/vessel_segmentation/8-unet/best_model.pth", weights_only=True))
post_pred = AsDiscrete(argmax=True)

label_vessel_amount = []
for i in range(len(ds)):
    print(i)

    tmp = ds[i]

    patient_id = tmp["pre"].split("/")[6]

    tmp = trans(tmp)

    img = tmp["post_2"]
    seg = tmp["vessels"]

    val_inputs = tmp["img"]
    with torch.no_grad():
        val_outputs = sliding_window_inference(val_inputs[None, ...], [96, 96, 96], 4, model)[0]
    seg = post_pred(val_outputs)
    # seg = val_outputs[:1]

    # M = compute_affine(orientation)
    # print(img.meta['00200037'])
    # print(img.meta['00200032'])
    # print(img.meta['spacing'])
    # print(img.meta['original_affine'])
    # print(img.meta['lastImagePositionPatient'])
    y1, y2, x1, x2, z1, z2 = bbs[bbs["Patient ID"] == patient_id].iloc[:, 1:].values[0]

    orientation = img.meta['00200037']["Value"]
    if orientation[0] < 0:
        print("transforming X")
        x1 = img.shape[1] - x1
        x2 = img.shape[1] - x2
        x1, x2 = x2, x1
    if orientation[4] < 0:
        print("transforming Y")
        y1 = img.shape[1] - y1
        y2 = img.shape[1] - y2
        y1, y2 = y2, y1

    num_cols = seg.shape[1]
    vessels_amount_left = seg[0, :num_cols//2, :, :].sum()
    vessels_amount_right = seg[0, num_cols//2:, :, :].sum()

    tot = vessels_amount_left + vessels_amount_right

    if x1 > img.shape[1]/2:
        right = True
    else:
        right = False

    if x2 < img.shape[1]/2:
        left = True
    else:
        left = False

    label_vessel_amount.append([vessels_amount_left.item(), vessels_amount_right.item(), int(left), int(right)])

    # slice = (z1+z2)//2
    # draw = cv.cvtColor(img.__array__()[0, :, :, slice], cv.COLOR_GRAY2BGR)
    # draw /= draw.max()
    # draw = cv.rectangle(draw, (y1, x1), (y2, x2), (0, 255, 0), 2)
    # plt.imshow(draw.transpose(1, 0, 2))
    # plt.title(f"Slice {slice}")
    # plt.show()
    #
    # draw = cv.cvtColor(img[0].max(2).values.__array__(), cv.COLOR_GRAY2BGR)
    # draw /= draw.max()
    # draw = cv.rectangle(draw, (y1, x1), (y2, x2), (0, 255, 0), 2)
    # plt.imshow(draw.transpose(1, 0, 2))
    # plt.title(f"MIP, left={vessels_amount_left/tot*100:.2f}%, right={vessels_amount_right/tot*100:.2f}%")
    # plt.show()

a = np.array(label_vessel_amount)
x = (a[:,0] > a[:, 1]).astype("int")
y = a[:, 2]
from sklearn.metrics import matthews_corrcoef
print(f"MCC={matthews_corrcoef(x, y)}")
#torch.save(label_vessel_amount,"/cache/marco/datasets/Duke-Breast-Cancer-MRI/other/label_vessel_amount")
