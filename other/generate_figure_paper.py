import torch
from monai.data import Dataset
import cv2 as cv
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImaged, Compose, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld
from monai.inferers import sliding_window_inference
import os
from config import *
import matplotlib.pyplot as plt

save_path = "/cache/marco/image_figure/seed34_test0/"
cfg_path = "/cache/marco/experiments/paper_experiments/seed34_AttentionUnet/config.yaml"
cfg = load_config(cfg_path)
cfg["TRAINING"]["cache_ratio"] = None
cfg["TRAINING"]["device"] = "cpu"

data = torch.load(cfg["TRAINING"]["monai_data"], weights_only=True)

if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
    root = cfg["TRAINING"]["dataset_root"]
    for elem in data:
        for key in elem:
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
    data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])
    factor = (len(data)+len(test_data))/len(data)
else:
    factor = 1

train_data, valid_data = train_test_split(data, test_size=factor*cfg["TRAINING"]["valid_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])

train_transforms = None
if "TRAIN_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TRAIN_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TRAIN_TRANSFORM"][i]))
    train_transforms = Compose(transform_list)

test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)

train_dataset = Dataset(train_data, train_transforms)
valid_dataset = Dataset(valid_data, test_transforms)
t = RandCropByPosNegLabeld(image_key="img", image_threshold=0, keys=["img", "seg"], label_key="seg", neg=1, num_samples=4, pos=3, spatial_size=[96, 96, 96])

sample = valid_dataset[0]

sample["img"][0] /= sample["img"][0].max()
sample["img"][1] /= sample["img"][1].max()

os.mkdir(os.path.join(save_path, "mip"))
cv.imwrite(os.path.join(save_path, "mip", "1_post.png"),(sample["img"][1].max(2).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "mip", "2_post.png"),(sample["img"][1].max(1).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "mip", "3_post.png"), (sample["img"][1].max(0).values * 255).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "mip", "1_pre.png"),(sample["img"][0].max(2).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "mip", "2_pre.png"),(sample["img"][0].max(1).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "mip", "3_pre.png"), (sample["img"][0].max(0).values * 255).numpy().astype("uint8"))

os.mkdir(os.path.join(save_path, "gt_mip_whole"))
cv.imwrite(os.path.join(save_path, "gt_mip_whole", "1.png"),(sample["seg"][0].max(2).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "gt_mip_whole", "2.png"),(sample["seg"][0].max(1).values * 255).transpose(1, 0).numpy().astype("uint8"))
cv.imwrite(os.path.join(save_path, "gt_mip_whole", "3.png"), (sample["seg"][0].max(0).values * 255).numpy().astype("uint8"))

sample_subvolumes = t(sample)

os.mkdir(os.path.join(save_path, "subvolumes_slice"))
for i, x in enumerate(sample_subvolumes):
    os.mkdir(os.path.join(save_path, "subvolumes_slice", str(i)))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "end_1_pre.png"),
               (x["img"][0][:, :, 95] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "start_2_pre.png"),
               (x["img"][0][:, 0, :] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "start_3_pre.png"),
               (x["img"][0][0, :, :] * 255).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "end_1_post.png"),
               (x["img"][1][:, :, 95] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "start_2_post.png"),
               (x["img"][1][:, 0, :] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "subvolumes_slice", str(i), "start_3_post.png"),
               (x["img"][1][0, :, :] * 255).numpy().astype("uint8"))


os.mkdir(os.path.join(save_path, "gt"))
for i, x in enumerate(sample_subvolumes):
    os.mkdir(os.path.join(save_path, "gt", str(i)))
    cv.imwrite(os.path.join(save_path, "gt", str(i), "end_1_pre.png"),
               (x["seg"][0][:, :, 95] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "gt", str(i), "start_2_pre.png"),
               (x["seg"][0][:, 0, :] * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "gt", str(i), "start_3_pre.png"),
               (x["seg"][0][0, :, :] * 255).numpy().astype("uint8"))

os.mkdir(os.path.join(save_path, "gt_mip"))
for i, x in enumerate(sample_subvolumes):
    os.mkdir(os.path.join(save_path, "gt_mip", str(i)))
    cv.imwrite(os.path.join(save_path, "gt_mip", str(i), "1.png"),
               (x["seg"][0].max(2).values * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "gt_mip", str(i), "2.png"),
               (x["seg"][0].max(1).values * 255).transpose(1, 0).numpy().astype("uint8"))
    cv.imwrite(os.path.join(save_path, "gt_mip", str(i), "3.png"),
               (x["seg"][0].max(0).values * 255).numpy().astype("uint8"))


device = torch.device(cfg["TRAINING"]["device"])
model_path = "/cache/marco/experiments/paper_experiments/seed34_AttentionUnet/best_model.pth"
model = create_object_from_dict(cfg["MODEL"]).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

img_list = [x["img"] for x in sample_subvolumes]
batch = torch.stack(img_list)

with torch.no_grad():
    val_outputs = sliding_window_inference(batch, [96, 96, 96], 4, model)

os.mkdir(os.path.join(save_path, "soft_output"))
for i, x in enumerate(val_outputs):
    os.mkdir(os.path.join(save_path, "soft_output", str(i)))
    # x[0] = (x[0]-x[0].min())/(x[0].max()-x[0].min())
    # x[1] = (x[1] - x[1].min()) / (x[1].max() - x[1].min())
    x = (x - x.min()) / (x.max() - x.min())
    tmp = x[0][:, :, 95]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "end_1_background.png"),
               (tmp * 255).transpose(1, 0).numpy().astype("uint8"))
    tmp = x[0][:, 0, :]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "start_2_background.png"),
               (tmp * 255).transpose(1, 0).numpy().astype("uint8"))
    tmp = x[0][0, :, :]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "start_3_background.png"),
               (tmp * 255).numpy().astype("uint8"))
    tmp = x[1][:, :, 95]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "end_1_vessels.png"),
               (tmp * 255).transpose(1, 0).numpy().astype("uint8"))
    tmp = x[1][:, 0, :]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "start_2_vessels.png"),
               (tmp * 255).transpose(1, 0).numpy().astype("uint8"))
    tmp = x[1][0, :, :]

    cv.imwrite(os.path.join(save_path, "soft_output", str(i), "start_3_vessels.png"),
               (tmp * 255).numpy().astype("uint8"))
