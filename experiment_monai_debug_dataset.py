from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImaged, Compose, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld
from customTransform import SliceOrderingd, ExtractVesselSegmentationd, Select_sequence
from monai.networks.nets import UNet
from train_functions import train_one_epoch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from math import ceil
import time
import copy
import os
import pydicom
from config import *
import argparse
import matplotlib.pyplot as plt
import copy

from monai.networks.nets import AttentionUnet


def test(test_loader, model, metric_fn, post_pred, post_label, device=torch.device("cpu"), print_per_epoch=5, roi_size=(96, 96, 96), sliding_window=True):
    model.eval()

    verbose_step = ceil(len(test_loader) / print_per_epoch)

    with torch.no_grad():
        start_time = time.time()
        for step_i, val_data in enumerate(test_loader):
            val_inputs, val_labels = (
                val_data["img"].to(device),
                val_data["seg"].to(device),
            )
            if "seg_mip" in val_data.keys():
                val_labels_mip = val_data["seg_mip"].to(device)

            if sliding_window:
                val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model)
            else:
                val_outputs = model(val_inputs)

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            if isinstance(metric_fn, list):
                val_labels_mip = [post_label(i) for i in decollate_batch(val_labels_mip)]
                metric_fn[0](y_pred=val_outputs, y=val_labels)
                metric_fn[1](y_pred=val_outputs, y=val_labels_mip)
            else:
                metric_fn(y_pred=val_outputs, y=val_labels)

            if step_i % verbose_step == verbose_step - 1:
                print(
                    f"test progress={100. * (step_i + 1) / len(test_loader):.1f}%\t"
                    f"elapsed time={time.time() - start_time:.3f} s")
                start_time = time.time()

        return metric_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading configuration at {args.config}")
    cfg = load_config(args.config)

    workspace = cfg["TRAINING"]["workspace"]

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

    with open(os.path.join(workspace, "train_split.txt"), "w") as f:
        for sample in train_data:
            f.write(sample["seg"].split("Segmentation_Masks_NRRD/")[1][:14]+"\n")
    with open(os.path.join(workspace, "valid_split.txt"), "w") as f:
        for sample in valid_data:
            f.write(sample["seg"].split("Segmentation_Masks_NRRD/")[1][:14]+"\n")
    if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
        with open(os.path.join(workspace, "test_split.txt"), "w") as f:
            for sample in test_data:
                f.write(sample["seg"].split("Segmentation_Masks_NRRD/")[1][:14] + "\n")

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

    if "cache_ratio" in cfg["TRAINING"].keys() and cfg["TRAINING"]["cache_ratio"] is not None:
        train_dataset = CacheDataset(train_data, train_transforms, cache_rate=cfg["TRAINING"]["cache_ratio"], num_workers=1)
        valid_dataset = CacheDataset(valid_data, test_transforms, cache_rate=cfg["TRAINING"]["cache_ratio"], num_workers=1)
    else:
        train_dataset = Dataset(train_data, train_transforms)
        valid_dataset = Dataset(valid_data, test_transforms)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=cfg["TRAINING"]["num_workers"])

    #train_loader = DataLoader(train_dataset, batch_size=cfg["TRAINING"]["train_batch_size"], shuffle=True, num_workers=cfg["TRAINING"]["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["TRAINING"]["test_batch_size"], shuffle=False, num_workers=cfg["TRAINING"]["num_workers"])

    if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
        test_dataset = Dataset(test_data, test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=cfg["TRAINING"]["test_batch_size"], shuffle=False,num_workers=cfg["TRAINING"]["num_workers"])


    b = train_dataset[37]

    #'/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/Duke-Breast-Cancer-MRI/Breast_MRI_681/1.3.6.1.4.1.14519.5.2.1.295695303595787910742187814712143877604/1.3.6.1.4.1.14519.5.2.1.83278429602180651460269499189988927799'