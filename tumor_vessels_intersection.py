from scipy.ndimage import binary_dilation
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
from torch.nn import Sigmoid
import os
import nrrd
import numpy as np
import cv2 as cv
from torch.nn import Sigmoid

import matplotlib.pyplot as plt

from monai.transforms import LoadImaged, Compose, ScaleIntensityd, Orientationd, AsDiscrete, Transposed
from customTransform import splitDCEd
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from config import create_object_from_dict, load_config


def tumor_surface_intersection(tumor_seg, vessel_seg, k_tumor, k_vessels):
    """
    Compute intersection of vessels on the surface of the tumor.

    Parameters:
        tumor_seg (np.ndarray): 3D binary array of tumor segmentation.
        vessel_seg (np.ndarray): 3D binary array of vessel segmentation.

    Returns:
        int: Count of intersecting voxels on the tumor surface.
        np.ndarray: Binary array marking intersection points.
    """
    # Ensure inputs are binary
    tumor_seg = (tumor_seg > 0).astype(np.uint8)
    vessel_seg = (vessel_seg > 0).astype(np.uint8)
    vessel_seg = cv.dilate(vessel_seg, np.ones((k_vessels, k_vessels), np.uint8), iterations=1)

    # Compute tumor surface: dilate and subtract the interior
    tumor_dilated = cv.dilate(tumor_seg, np.ones((k_tumor, k_tumor), np.uint8), iterations=1)
    tumor_surface = np.where(tumor_seg==1, 0, tumor_dilated)

    # Find intersection of vessels with tumor surface
    intersection = (tumor_surface & vessel_seg).astype(np.uint8)

    return intersection


def vessels_presence_around_tumor(tumor_seg, vessel_seg, dilate_tumor, normalize=False):
    tumor_seg = (tumor_seg > 0).astype(np.uint8)
    tumor_dilated = cv.dilate(tumor_seg, np.ones((dilate_tumor, dilate_tumor), np.uint8), iterations=1)
    tumor_surface = np.where(tumor_seg==1, 0, tumor_dilated)

    intersection = np.multiply(tumor_surface, vessel_seg)

    if normalize:
        intersection /= vessel_seg.sum()

    return intersection


def tumor_vessels_intersection(tumor_seg, vessel_seg, k_tumor, k_vessels):
    tumor_seg = (tumor_seg > 0).astype(np.uint8)
    tumor_seg = cv.dilate(tumor_seg, np.ones((k_tumor, k_tumor), np.uint8), iterations=1)
    vessel_seg = (vessel_seg > 0).astype(np.uint8)
    vessel_seg = cv.dilate(vessel_seg, np.ones((k_vessels, k_vessels), np.uint8), iterations=1)

    intersection = (tumor_seg & vessel_seg).astype(np.uint8)

    return intersection



dict = {
    0: "benign",
    1: "malignant"
}

if __name__ == "__main__":

    use_model = False
    prediction_folder = "/cache/marco/experiments/vessel_segmentation/6-attention_unet/AMBL_soft_predictions/"

    monai_data_path = "/cache/marco/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels"
    root = "/cache/marco/datasets/Advanced-MRI-Breast-Lesions/Advanced-MRI-Breast-Lesions/"
    cfg_path = "/cache/marco/experiments/vessel_segmentation/6-attention_unet/config.yaml"
    weights_path = "/cache/marco/experiments/vessel_segmentation/6-attention_unet/best_model.pth"

    # monai_data_path = r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\data_monai_AMBL"
    # root = r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\Advanced-MRI-Breast-Lesions"
    # cfg_path = r"C:\Users\marco\Desktop\MRI\8-unet\config.yaml"
    # weights_path = r"C:\Users\marco\Desktop\MRI\8-unet\best_model.pth"

    data = torch.load(monai_data_path, weights_only=False)
    for elem in data:
        for key in elem:
            if isinstance(elem[key], str):
                elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

    if use_model:
        cfg = load_config(cfg_path)
        device = torch.device("cpu")
        model = create_object_from_dict(cfg["MODEL"])
        model = model.to(device)
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
        model.eval()

    if use_model:
        transform = Compose([LoadImaged(keys=["img", "seg"]),
                         splitDCEd("img", 5, [0, 2]),
                         Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                         Orientationd(keys=["img", "seg"], axcodes="LPS"),
                         ScaleIntensityd(keys=["img"])])
    else:
        transform = Compose([LoadImaged(keys=["seg"]),
                             Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                             Orientationd(keys=["seg"], axcodes="LPS")])

    dataset = Dataset(data)



    res = {}

    labels = []
    intersections = []

    for i in range(len(dataset)):
        print(i)
        sample = dataset[i]
        if not use_model:
            id = sample["img"].split("/")[6]
        # print(f'image {id}, ', end="")
        sample = transform(sample)

        if use_model:
            with torch.no_grad():
                soft = sliding_window_inference(sample["img"][None, ...].to(device), [96, 96, 96], 4, model)[0].cpu().numpy()
                hard = AsDiscrete(argmax=True)(soft)[0].numpy().astype(np.uint8)
        else:
            hard = nrrd.read(os.path.join("/cache/marco/experiments/vessel_segmentation/6-attention_unet/inference_AMBL/", id+".nrrd"))[0]
            soft = nrrd.read(os.path.join(prediction_folder, id+".nrrd"))[0]

        for j in range(len(sample["label"])):
            # print(f'lesion {j} -> ', end="")
            if sample["label"][j] is None:
                continue
            tumor = sample["seg"][j].numpy().astype(np.uint8)

            blank = np.zeros_like(tumor)

            if sample["label"][j] == 1:
                h = np.stack([hard, blank, tumor])
                s = np.stack([soft, blank, tumor])
            else:
                h = np.stack([hard, tumor, blank])
                s = np.stack([soft, tumor, blank])


            print(f"{id}_{j}_{dict[sample['label'][j]]}_hard.png")

            cv.imwrite(os.path.join("/cache/marco/experiments/vessel_segmentation/6-attention_unet/a/", f"{id}_{j}_{dict[sample['label'][j]]}_hard.png"), (np.max(h, 3).transpose(2, 1, 0)*255).astype("uint8"))
            # cv.imwrite(os.path.join("/cache/marco/experiments/vessel_segmentation/6-attention_unet/a/", f"{id}_{j}_{dict[sample['label'][j]]}_soft.png"), (np.max(s, 3).transpose(2, 1, 0)*255).astype("uint8"))


    #         intersection = vessels_presence_around_tumor(tumor, soft, 15, normalize=False)
    #
    #         print(f'intersection={intersection.sum()/soft.sum()} label={dict[sample["label"][j]]}\n')
    #
    #         intersections.append(intersection.sum())
    #         labels.append(sample["label"][j])
    #
    # results_path = "/cache/marco/experiments/lesion_classification/AttentionUnet_soft_normalize_vessels/"
    #
    # tpr, fpr, ths = roc_curve(labels, intersections)
    # auc = roc_auc_score(labels, intersections)
    #
    # fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    # ax.plot(tpr, fpr, label=f'AUC={auc*100:.2f}')
    # ax.grid(True)
    # ax.legend()
    # fig.savefig(os.path.join(results_path, "roc.png"))
    # plt.close(fig)
    #
    # with open(os.path.join(results_path, "labels.txt"), "w") as f:
    #     for i in range(len(labels)):
    #         f.write(f'{labels[i]}\n')
    #
    # with open(os.path.join(results_path, "score.txt"), "w") as f:
    #     for i in range(len(intersections)):
    #         f.write(f'{intersections[i]}\n')
