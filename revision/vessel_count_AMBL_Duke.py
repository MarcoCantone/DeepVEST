import torch
import os

import matplotlib.pyplot as plt

import nrrd
import pydicom
from torch.nn import Sigmoid

from config import load_config, create_object_from_dict
from monai.inferers import sliding_window_inference
from monai.data import Dataset, PydicomReader, decollate_batch
from monai.transforms import LoadImaged, Compose, ScaleIntensityd, Orientationd, AsDiscrete, Transposed
from monai.metrics import ConfusionMatrixMetric
from customTransform import splitDCEd
from sklearn.model_selection import train_test_split

AMBL = False
Duke = True


if AMBL:
    data = torch.load("/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels", weights_only=False)
    load_seg_mip = True
    root = "/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/Advanced-MRI-Breast-Lesions/"
    root_mip = "/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/"
    for elem in data:
        for key in elem:
            if isinstance(elem[key], str):
                elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None
        if load_seg_mip:
            id = elem["seg"].split("/")[6]
            mip_relative_path = os.path.join("MIP_post_annotation", id, "Vessels.seg.nrrd")
            if os.path.exists(os.path.join(root_mip, mip_relative_path)):
                elem["seg_mip"] = os.path.join(root_mip, mip_relative_path)
            else:
                elem["seg_mip"] = None

    transform = Compose([LoadImaged(keys=["seg_mip"], reader=PydicomReader())])

    data = [elem for elem in data if elem["seg_mip"] is not None]
    dataset = Dataset(data, transform=transform)

    vessel_count = 0
    voxel_count = 0

    for elem in dataset:
        mip = elem["seg_mip"]

        vessel_count += (mip==1).sum()
        voxel_count += mip.numel()

        print(vessel_count, voxel_count)

    print(vessel_count/voxel_count)







if Duke:
    dataset_root = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
    folder = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/"
    cfg = load_config(os.path.join(folder, "config.yaml"))
    monai_metadata = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative"

    data = torch.load(monai_metadata)

    load_seg_mip = True
    if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
        # root = cfg["TRAINING"]["dataset_root"]
        for elem in data:
            if load_seg_mip:
                id = elem["seg"].split("/")[1]
                mip_relative_path = os.path.join("MIP post annotation", id, "vessels.seg.nrrd")
                if os.path.exists(os.path.join(dataset_root, mip_relative_path)):
                    elem["seg_mip"] = mip_relative_path
                else:
                    elem["seg_mip"] = None
            for key in elem:
                if isinstance(elem[key], str):
                    elem[key] = os.path.join(dataset_root, elem[key]) if elem[key] is not None else None

    if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
        train_data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"],
                                                 random_state=cfg["TRAINING"]["train_test_split_seed"])
    else:
        train_data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["valid_size"],
                                             random_state=cfg["TRAINING"]["train_test_split_seed"])

    transform = Compose([LoadImaged(keys=["seg_mip"], reader=PydicomReader())])

    dataset = Dataset(test_data, transform=transform)

    vessel_count = 0
    voxel_count = 0

    for elem in dataset:
        mip = elem["seg_mip"]

        vessel_count += (mip == 1).sum()
        voxel_count += mip.numel()

        print(vessel_count, voxel_count)

    print(vessel_count / voxel_count)