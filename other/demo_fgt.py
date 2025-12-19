import os
import random

from customTransform import nrrd_to_vesselMap
import nrrd
from utilities import load3dfrom2dslice_sitk
import cv2
import numpy as np
import torch

from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, AsDiscrete, EnsureChannelFirstd, ScaleIntensityd, Orientationd
from customTransform import SliceOrderingd, ExtractVesselSegmentationd
from config import load_config, create_object_from_dict
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split

folder = "C:/Users/marco/Desktop/MRI/99-unet_vessels_fgt"

cfg_path = os.path.join(folder, "config.yaml")
cfg = load_config(cfg_path)

data = torch.load("C:/Users/marco/Desktop/MRI/duke/data_monai_relative")
cfg["TRAINING"]["dataset_root"] = "C:/Users/marco/Desktop/MRI/duke"
if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
    root = cfg["TRAINING"]["dataset_root"]
    for elem in data:
        for key in elem:
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

_, data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])
# root_linux = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
# root_windows = "C:/Users/marco/Desktop/MRI/duke/"
# downloaded_ids = os.listdir("C:/Users/marco/Desktop/MRI/duke/Duke-Breast-Cancer-MRI")
# data = [{"pre": x["pre"].replace(root_linux, root_windows), "post_1": x["post_1"].replace(root_linux, root_windows), "seg": x["seg"].replace(root_linux, root_windows)} for x in data if x["seg"].split("/")[6] in downloaded_ids]

# create transform
test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)

dataset = Dataset(data, test_transforms)

# create model
weights_path = os.path.join(folder, "best_model.pth")
model = create_object_from_dict(cfg["MODEL"])
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")))
model.eval()

roi_size = [96, 96, 96]
sw_batch_size = 16

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])

sample = dataset[14]

img = sample["img"]
seg = sample["seg"]

with torch.no_grad():
    pred = sliding_window_inference(img[None, ...], roi_size, sw_batch_size, model)
    pred = [post_pred(i) for i in decollate_batch(pred)]

vessels = pred[0][1].numpy()
dense = pred[0][2].numpy()
post = img[1].numpy()

kernel = np.ones((3, 3), np.uint8)
vessels = cv2.dilate(vessels, kernel, iterations=1)

post_bgr = post[...,None].repeat(3, 3)

slice_seg = np.zeros(post_bgr.shape)
slice_seg[:, :, :, 1] = dense
slice_seg[:, :, :, 2] = vessels

segment = False

def update_slice(val):
    global current_slice
    current_slice = val
    update_image()

def switch_mode(event, x, y, flags, param):
    global segment
    if event == cv2.EVENT_LBUTTONDOWN:
        segment = not segment
        update_image()

def update_image():
    slice = post_bgr[:, :, current_slice, :]
    if segment:

        slice = slice + 0.25 * slice_seg[:, :, current_slice, :]

    cv2.imshow("win", slice.transpose(1, 0, 2))


cv2.namedWindow("win", cv2.WINDOW_NORMAL)

cv2.createTrackbar("tk", "win", 0, img.shape[-1]-1, update_slice)

cv2.setMouseCallback("win", switch_mode)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows
cv2.destroyAllWindows()