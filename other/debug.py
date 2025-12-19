import torch

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

from utilities import generate_monai_AMBL

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

device = "cpu"

model = create_object_from_dict(cfg["MODEL"]).to(device)
model.load_state_dict(torch.load("/cache/marco/experiments/vessel_segmentation/8-unet/best_model.pth", weights_only=True, map_location=device))

step = 52

sample = test_dataset[0]
img = sample["img"].to(device)
img = img[None, ...]
val_label = sample["seg"]
# img.shape [B, C, H, W, D]
unfolded_img = img.unfold(2, 96, step).unfold(3, 96, step).unfold(4, 96, step)
# img.shape [B, C, n_h, n_w, n_d, 96, 96, 96]
num_subvol = np.prod(unfolded_img.shape[2:5])
batch_size = unfolded_img.shape[0]
reshaped = unfolded_img.reshape(list(unfolded_img.shape[:2]) + [num_subvol]+list(unfolded_img.shape[5:]))
# [B, C, n_h * n_w * n_d, 96, 96, 96]
reshaped = reshaped.permute(0, 2, 1, 3, 4, 5)
# [B, n_h * n_w * n_d, C, 96, 96, 96]
reshaped = reshaped.view(reshaped.shape[0]*reshaped.shape[1], *reshaped.shape[2:])
# [B * n_h * n_w * n_d, C, 96, 96, 96]
output = model(reshaped)
# [B * n_h * n_w * n_d, C, 96, 96, 96]
output = output.view(batch_size, num_subvol, *output.shape[1:])
# [B, n_h * n_w * n_d, C, 96, 96, 96]
output = output.permute(0, 2, 1, 3, 4, 5)
# [B, C, n_h * n_w * n_d, 96, 96, 96]
output_reshaped = output.reshape(unfolded_img.shape)

#test = output_reshaped.permute(0, 1, 4, 2, 5, 3, 6)
#prediction = test.reshape(img.shape)


reconstructed = torch.zeros_like(img)
overlap_count = torch.zeros_like(img)
for b in range(output_reshaped.shape[0]):
    for i in range(output_reshaped.shape[2]):  # Patches along H
        for j in range(output_reshaped.shape[3]):  # Patches along W
            for k in range(output_reshaped.shape[4]):  # Patches along D
                h_start = i * step
                w_start = j * step
                d_start = k * step
                # print(b, i, j, k)

                # Add the patch to the reconstructed image
                reconstructed[
                    b,
                    :,
                    h_start:h_start + 96,
                    w_start:w_start + 96,
                    d_start:d_start + 96,
                ] += output_reshaped[b, :, i, j, k]

                # Count overlaps
                overlap_count[
                    b,
                    :,
                    h_start:h_start + 96,
                    w_start:w_start + 96,
                    d_start:d_start + 96,
                ] += 1



# Normalize overlapping regions
prediction = reconstructed / overlap_count

boundaries = step*np.array(unfolded_img.shape[2:5])+(96-step)

prediction = prediction[:, :, :boundaries[0], :boundaries[1], :boundaries[2]]
val_label = val_label[:, :boundaries[0], :boundaries[1], :boundaries[2]]

prediction_2d = prediction[0, 1].max(2).values
val_label_2d = val_label[0].max(2).values

from monai.losses import DiceLoss

loss = DiceLoss(sigmoid=True)(prediction_2d[None, None], val_label_2d[None, None])

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(torch.nn.Sigmoid()(prediction_2d.permute(1, 0).detach()), cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(val_label_2d.permute(1, 0).detach(), cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(img[0, 1].max(2).values.permute(1, 0).detach(), cmap='gray')
plt.show()


trans2 = Compose([
    LoadImaged(keys=['pre', 'post_2', 'vessels_mip', 'seg']),
    EnsureChannelFirstd(keys=['pre', 'post_2', "vessels_mip", 'seg']),
    ConcatItemsd(keys=['pre', 'post_2'], name="img", dim=0),
    SliceOrderingd(img_key='img'),
    separate_vessels_breast(seg_key='seg', channel=True, compute_backgroud=False, dense=False, adjust_affine="img"),
    ConcatItemsd(keys=['vessels'], name="seg", dim=0),
    Orientationd(keys=["img", "seg", "vessels_mip"], axcodes="LPS"),
    ScaleIntensityd(keys=["img"], maxv=1.0, minv=0.0),
    DeleteItemsd(keys=["pre", "post_1", "post_2", "post_3", "post_4", "breast"]),
])
# sample = trans2(dataset[0])
# for i in range(len(dataset)):
#     sample = trans2(dataset[i])
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(sample["img"][1].max(2).values.permute(1, 0), cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.imshow(sample["vessels_mip"][0, :, :, 0].permute(1, 0), cmap="gray")
#     plt.subplot(1, 3, 3)
#     plt.imshow(sample["seg"][0].max(2).values.permute(1, 0), cmap="gray")
#     plt.show()

# ids = sorted(os.listdir("/data/cantone/datasets/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI/"))
# ids.remove("LICENSE")
#
# df = path_seq_map("/data/cantone/datasets/Duke-Breast-Cancer-MRI/Breast-Cancer-MRI-filepath_filename-mapping.csv")
# root = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
#
# bb = pd.read_csv("/data/cantone/datasets/Duke-Breast-Cancer-MRI/Annotation_Boxes.csv")
#
# new_bb = []
#
# for id in ids:
#     mri_volume = LoadImage()(os.path.join(root, df[(df['sequence'] == 'post_2') & (df['classic_path'].str.contains(id))]["classic_path"].item()))
#     row = bb[bb["Patient ID"] == id]
#     p1 = np.array([row["Start Column"].item(), row["Start Row"].item(), row["Start Slice"].item(), 1])
#     p2 = np.array([row["End Column"].item(), row["End Row"].item(), row["End Slice"].item(), 1])
#     s_col, s_row, s_slice, _ = mri_volume.meta["original_affine"] @ p1
#     e_col, e_row, e_slice, _ = mri_volume.meta["original_affine"] @ p2
#     new_bb.append([id, s_row, e_row, s_col, e_col, s_slice, e_slice])
#
# new_bb = pd.DataFrame(new_bb, columns=bb.columns)
# new_bb.to_csv("/data/cantone/datasets/Duke-Breast-Cancer-MRI/Annotation_Boxes_patient_space.csv", index=False)

# ds = Dataset(data)
#
# trans = Compose([
#     LoadImaged(keys=['pre', 'post_2', 'seg']),
#     EnsureChannelFirstd(keys=['pre', 'post_2', "seg"]),
#     ConcatItemsd(keys=['pre', 'post_2'], name="img", dim=0),
#     SliceOrderingd(img_key='img'),
#     separate_vessels_breast(seg_key='seg', channel=True, compute_backgroud=True),
#     ConcatItemsd(keys=['background', 'vessels', 'dense'], name="seg", dim=0),
#     ScaleIntensityd(keys=["img"], maxv=1.0, minv=0.0),
#     AsDiscreted(keys=['seg'], argmax=True)
# ])
#
# trans2 = Compose([
#     LoadImaged(keys=['pre', 'post_2', 'seg']),
#     EnsureChannelFirstd(keys=['pre', 'post_2', "seg"]),
#     ConcatItemsd(keys=['pre', 'post_2'], name="img", dim=0),
#     SliceOrderingd(img_key='img'),
#     separate_vessels_breast(seg_key='seg', channel=True, compute_backgroud=True, adjust_affine="img"),
#     ConcatItemsd(keys=['background', 'vessels', 'dense'], name="seg", dim=0),
#     Orientationd(keys=["img", "seg"], axcodes="LPS"),
#     ScaleIntensityd(keys=["img"], maxv=1.0, minv=0.0),
#     AsDiscreted(keys=['seg'], argmax=True)
# ])
#
# t3 = Compose([
#     LoadImaged(keys=['pre', 'post_2', 'seg']),
#     EnsureChannelFirstd(keys=['pre', 'post_2', "seg"]),
#     ConcatItemsd(keys=['pre', 'post_2'], name="img", dim=0),
#     SliceOrderingd(img_key='img'),
#     ScaleIntensityd(keys=["img"], maxv=1.0, minv=0.0),
#     separate_vessels_breast(seg_key='seg', channel=True, vessels=True, dense=False, compute_backgroud=False, adjust_affine="img"),
#     ConcatItemsd(keys=['vessels'], name="seg", dim=0),
#     DeleteItemsd(keys=["pre", "post_1", "post_2", "post_3", "post_4", "vessels", "breast"]),
#     Orientationd(keys=["img", "seg"], axcodes="LPS"),
#     RandCropByPosNegLabeld(keys=["img", "seg"], image_key="img", label_key="seg", neg=1, pos=3, num_samples=4, spatial_size=[96, 96, 96]),
#     RandAffined(keys=["img", "seg"], prob=1, rotate_range=[np.pi/8, np.pi/8, np.pi/8], shear_range=[0.1, 0.1, 0.1], scale_range=[0.1, 0.1, 0.1], translate_range=[10, 10, 10]),
#     RandFlipd(keys=["img", "seg"], prob=1, spatial_axis=0),
#     RandFlipd(keys=["img", "seg"], prob=1, spatial_axis=1),
#     RandFlipd(keys=["img", "seg"], prob=1, spatial_axis=2)
# ])
#
# randcrop = RandSpatialCropSamplesd(keys=["img", "seg"], roi_size=[96, 96, 96], num_samples=4)
# randcrop_scale = RandSpatialCropSamples([48, 48, 48], 4, max_roi_size=[96, 96, 96], random_size=True)
# resize = Resize(spatial_size=[96, 96, 96])
# randcrop = RandCropByPosNegLabeld(keys=["img", "seg"], image_key="img", label_key="seg", neg=1, pos=3, num_samples=4, spatial_size=[96, 96, 96])
#
# for i in range(5):
#     sample = trans2(ds[random.randint(0, 98)])
#     plt.subplot(1, 2, 1)
#     plt.imshow(sample["img"][1, :, :, 70].permute(1, 0), cmap="gray")
#     plt.subplot(1, 2, 2)
#     plt.imshow(sample["seg"][0, :, :, 70].permute(1, 0), cmap="gray")
#     plt.show()
