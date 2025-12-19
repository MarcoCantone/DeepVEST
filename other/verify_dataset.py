from customTransform import *
from customDatasets import DukeMRI_Seg_MS
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2 as cv
import random
import os
import json

from monai.data import CacheDataset, Dataset
from monai.transforms import Orientationd, EnsureChannelFirstd

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

t = Compose([
    LoadImaged(keys=['pre', 'post_1', "seg"]),
    EnsureChannelFirstd(keys=['pre', 'post_1']),
    Orientationd(keys=['pre', 'post_1'], axcodes='LPS'),
])

t1 = Compose([
    LoadImaged(keys=['pre', 'post_1', "seg"]),
    Multi_sequence(keys=['pre', 'post_1']),
    EnsureChannelFirstd(keys=['seg']),
    Orientationd(keys=['img', 'seg'], axcodes='LPS'),
    ExtractVesselSegmentationd(seg_key='seg'),
    ScaleIntensityd(keys=['img', 'seg']),
    RandCropByPosNegLabeld(
        keys=["img", "seg"],
        label_key="seg",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=8,
        image_key="img",
        image_threshold=0)
])

t2 = Compose([
    LoadImaged(keys=['pre', 'post_1', "seg"]),
    ExtractVesselSegmentationd_new(seg_key='seg'),
    Multi_sequence(keys=['pre', 'post_1']),
    EnsureChannelFirstd(keys=['seg']),
    Orientationd(keys=['img', 'seg'], axcodes='LPS'),
    ScaleIntensityd(keys=['img', 'seg']),
])



aug = Compose([
RandFlipd(
            keys=["img", "seg"],
            spatial_axis=[0],
            prob=0.15,
        ),
        RandFlipd(
            keys=["img", "seg"],
            spatial_axis=[1],
            prob=0.15,
        ),
        RandFlipd(
            keys=["img", "seg"],
            spatial_axis=[2],
            prob=0.15,
        ),
        RandRotate90d(
            keys=["img", "seg"],
            prob=0.15,
            max_k=3,
        )
])

data = torch.load("/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai")

# dataset = CacheDataset(
#     data=torch.load("/data/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_post1"),
#     transform=t,
#     cache_rate=1.0,
#     num_workers=32
# )

t3 = Compose([
    LoadImaged(keys=['pre', 'post_1', "seg"]),
    ExtractVesselSegmentationd_new(seg_key='seg'),
    Multi_sequence(keys=['pre', 'post_1']),
    EnsureChannelFirstd(keys=['seg']),
])

t4 = Compose([
    LoadImaged(keys=['pre', 'post_1', "seg"]),
    Multi_sequence(keys=['pre', 'post_1']),
    EnsureChannelFirstd(keys=['seg']),
    ExtractVesselSegmentationd(seg_key='seg'),
    SliceOrderingd(img_key='img'),
    ScaleIntensityd(keys=['img', 'seg']),

])

t5 = Compose([t4, aug])

dataset = Dataset(data=data, transform=t5)

i = 11
sample = dataset[i]
plt.subplot(1, 2, 1)
plt.imshow(sample["img"][1, :, :, 70].permute(1, 0), cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(sample["seg"][0, :, :, 70].permute(1, 0), cmap="gray")
plt.show()

i = 7
sample = dataset[i]
aug_sample = aug(sample)
for img_to_show in [sample, aug_sample]:
    plt.figure(figsize=(20, 5))
    for i in range(len(img_to_show)):
        plt.subplot(2, len(img_to_show), i + 1)
        plt.imshow(img_to_show[i]["img"][1, :, :, 48].permute(1, 0), cmap='gray')
        plt.subplot(2, len(img_to_show), len(img_to_show) + i + 1)
        plt.imshow(img_to_show[i]["seg"][0, :, :, 48].permute(1, 0), cmap='gray')
    plt.show()
    plt.clf()


i = 5
sample = dataset[i]
img = sample["post_1"]
meta = img.meta
print(meta['00200037'])
plt.imshow(img[0, :, :, 70].transpose(1, 0), cmap='gray')
plt.show()

# # for i in range(len(dataset)):
# #     print(i)
# #     sample = dataset[i]
# #     plt.figure(figsize=(10, 5))
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(sample['img'][0, :, :, 40], cmap="gray")
# #     plt.subplot(1, 2, 2)
# #     plt.imshow(sample['seg'][0, :, :, 40], cmap="gray")
# #     plt.savefig("/home/cantone/tmp/" + str(i) + ".png")
# #
# # for sample in dataset:
# #     cv.imshow("sample", sample["img"][0, :, :, 50])
# #
# #
# # img_dir = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
# # label_dir = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/Segmentation_Masks_NRRD/"
# # data_info = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/other/metadata_post1"
# # data_t = transforms.Compose([normalize_image(), MinMax(), np2tensor()])
# # label_t = nrrd_to_vesselMap()
# # dataset = DukeMRI_Seg_MS(img_dir, data_info, data_t, label_t, True)
#
# a, b = dataset.random_split()
#
# sample, label = a[0]
#
# root = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
#
#
# img_size_d = {}
# label_size_d = {}
#
# joint_d = {}
#
#
# # verify label and img shape
# # for idx in range(len(dataset)):
# #     print(dataset.ids[idx])
# #     series, label = dataset[idx]
# #
# #     img = series["post_1"]
# #
# #     key = str(img.shape)
# #     if key not in img_size_d.keys():
# #         img_size_d[key] = 1
# #     else:
# #         img_size_d[key] += 1
# #
# #     key = str(label.shape)
# #     if key not in label_size_d.keys():
# #         label_size_d[key] = 1
# #     else:
# #         label_size_d[key] += 1
# #
# #     key = str(label.shape+img.shape)
# #     if key not in joint_d.keys():
# #         joint_d[key] = 1
# #     else:
# #         joint_d[key] += 1
# #
# #     print(f"img shape:{img.shape}, label shape:{label.shape}")
# #     if img.shape != label.shape:
# #         print(f"DIFFERENT SHAPE FOR {dataset.ids[idx]}")
# #
# #     print("\n")
# #
# # with open("/home/cantone/img_size_d.json", "w") as f:
# #     f.write(json.dumps(img_size_d))
# #
# # with open("/home/cantone/label_size_d.json", "w") as f:
# #     f.write(json.dumps(label_size_d))
# #
# # with open("/home/cantone/joint_d.json", "w") as f:
# #     f.write(json.dumps(joint_d))
#
# #plot
# for idx in range(len(dataset)):
#     if dataset.ids[idx] in ["Breast_MRI_246", "Breast_MRI_435"]:
#         continue
#     print(idx)
#     series, label = dataset[idx]
#     img = series["post_1"]
#
#     slice = len(img)//2
#
#     draw = cv.cvtColor(img[slice].astype("float32"), cv.COLOR_GRAY2BGR)
#     draw = (draw - draw.min()) / (draw.max() - draw.min())
#
#     red_mask = np.zeros_like(draw).astype("float")
#     red_mask[:, :, 2] = 1.0
#
#     overlay = np.where(label[slice][:, :, None] == 1, red_mask, draw)
#
#     cv.imwrite(os.path.join("/home/cantone/tmp/", dataset.ids[idx]+".png"), overlay*255)
#
#     # num = 10
#     # indices = np.linspace(0, len(img)-1, num).astype("int")
#     # plt.figure(figsize=(50, 15))
#     # for i in range(num):
#     #     print(i)
#     #     plt.subplot(3, num, i + 1)
#     #     plt.imshow(img[indices[i]], cmap="gray")
#     #     plt.axis("off")
#     #
#     #     plt.subplot(3, num, num + i + 1)
#     #     plt.imshow(label[indices[i]], cmap="gray")
#     #     plt.axis("off")
#     #
#     #     draw = cv.cvtColor(img[indices[i]].astype("float32"), cv.COLOR_GRAY2BGR)
#     #     draw = (draw - draw.min()) / (draw.max() - draw.min())
#     #
#     #     red_mask = np.zeros_like(draw).astype("float")
#     #     red_mask[:, :, 0] = 1.0
#     #
#     #     overlay = np.where(label[indices[i]][:, :, None] == 1, red_mask, draw)
#     #
#     #     plt.subplot(3, num, 2 * num + i + 1)
#     #     plt.imshow(overlay, cmap="gray")
#     #     plt.axis("off")
#     #
#     # plt.show()
