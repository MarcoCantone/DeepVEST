import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import torch

from monai.data import Dataset, decollate_batch
from monai.transforms import Compose
from monai.inferers import sliding_window_inference

from skimage.morphology import octahedron
from scipy.ndimage import grey_dilation
from sklearn.model_selection import train_test_split

from config import load_config, create_object_from_dict


def vessels_removal_hard(mri, hard_prediction, dilation_size):
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_pred = cv.dilate(hard_prediction, kernel, iterations=1)
    post_no_vessels = np.where(dilated_pred == 1, 0, mri)
    return np.max(post_no_vessels, axis=2)


def vessels_removal_soft(mri, soft_prediction, d = 0, thresh = 0, strong = False):
    if d > 1:
        if d%2==0:
            structuring_element = octahedron(d//2)
        else:
            structuring_element = np.ones((d, d, d), np.uint8)
        dilated_image = grey_dilation(soft_prediction, footprint=structuring_element)
    else:
        dilated_image = soft_prediction

    # hard removal
    if strong:
        post_no_vessels = np.where(dilated_image > thresh, 0, mri)
    else:
        post_no_vessels = np.where(soft_prediction > thresh, 0, mri)

    # soft removal
    post_no_vessels = post_no_vessels*(1-dilated_image)

    # MIP and return
    return np.max(post_no_vessels, axis=2)


def vessels_removal_soft2(mri, soft_prediction, d = 0, thresh = 0.2, scale=60):
    if d > 1:
        if d%2==0:
            structuring_element = octahedron(d//2)
        else:
            structuring_element = np.ones((d, d, d), np.uint8)
        dilated_image = grey_dilation(soft_prediction, footprint=structuring_element)
    else:
        dilated_image = soft_prediction

    # soft removal
    post_no_vessels = mri * 1 - sigmoid_fn((dilated_image - thresh) * scale)

    # MIP and return
    return np.max(post_no_vessels, axis=2)

def sigmoid_fn(z):
    return 1/(1 + np.exp(-z))

# def vessels_removal_soft2(mri, soft_prediction, thresh):
#     post_no_vessels = np.where(soft_prediction > thresh, 0, mri * (1-(1/thresh)*soft_prediction))
#
#     # MIP and return
#     return np.max(post_no_vessels, axis=2)
#
# def vessels_removal_soft3(mri, soft_prediction, thresh):
#     # FIXME: using np.where is wrong
#     post_no_vessels = np.where(soft_prediction > thresh, 0, mri * 1-sigmoid_fn((soft_prediction-thresh)*60))
#
#     # MIP and return
#     return np.max(post_no_vessels, axis=2)

if __name__ == "main":

    device = torch.device("cuda:0")
    roi_size = [96, 96, 96]
    sw_batch_size = 8

    cfg_path = "/cache/marco/experiments/vessel_segmentation/9997-AttentionUnet_mip/config.yaml"
    model_name = "best_model.pth"
    channel = 1
    channel_post = 1
    sigmoid = True

    models = [
        (
            "/cache/marco/experiments/vessel_segmentation/9997-AttentionUnet_mip/config.yaml",
            "best_model.pth",
            1

        ),
        (
            "/cache/marco/experiments/vessel_segmentation/9997-AttentionUnet_mip/config.yaml",
            "best_model_mip.pth",
            1
        ),
        (
            "/cache/marco/experiments/vessel_segmentation/9999-Attention_onlyPost2/config.yaml",
            "best_model_mip.pth",
            0
        )
    ]

    for sample_id in range(20):
    # for sample_id in [18]:
        plt.figure(figsize=(29.7, 21))
        for j, (cfg_path, model_name, channel_post) in enumerate(models):
            cfg = load_config(cfg_path)

            data = torch.load(cfg["TRAINING"]["monai_data"], weights_only=True)

            if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
                root = cfg["TRAINING"]["dataset_root"]
                for elem in data:
                    for key in elem:
                        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

            _, data = train_test_split(data, test_size=cfg["TRAINING"]["valid_size"],
                                       random_state=cfg["TRAINING"]["train_test_split_seed"])
            weights_path = os.path.join(cfg["TRAINING"]["workspace"], model_name)
            model = create_object_from_dict(cfg["MODEL"])
            model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
            model = model.to(device)
            model.eval()

            post_pred = None
            if "POST_PRED" in cfg.keys():
                transform_list = []
                for i in range(len(cfg["POST_PRED"])):
                    transform_list.append(create_object_from_dict(cfg["POST_PRED"][i]))
                post_pred = Compose(transform_list)
            test_transforms = None
            if "TEST_TRANSFORM" in cfg.keys():
                transform_list = []
                for i in range(len(cfg["TEST_TRANSFORM"])):
                    transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
                test_transforms = Compose(transform_list)

            dataset = Dataset(data)

            print(sample_id)
            sample = dataset[sample_id]
            id = sample["post_2"].split("/")[6]
            sample = test_transforms(sample)
            img = sample["img"]
            post = img[channel_post].numpy()

            img = img.to(device)
            print("inference")
            with torch.no_grad():
                soft_pred = sliding_window_inference(img[None, ...], roi_size, sw_batch_size, model)
                hard_pred = [post_pred(i) for i in decollate_batch(soft_pred)]
                if channel==1:
                    soft_pred[0][channel] = soft_pred[0][1]-soft_pred[0][0]
                if sigmoid:
                    soft_pred = torch.nn.Sigmoid()(soft_pred[0][channel].cpu()).numpy()
                else:
                    soft_pred = soft_pred[0][channel].cpu().numpy()
                hard_pred = hard_pred[0][channel].cpu().numpy()

            mip = np.max(post, axis=2)
            pred_mip_hard = np.max(hard_pred, 2)
            pred_mip_soft = np.max(soft_pred, 2)

            # mip_no_vessels_hard = vessels_removal_hard(post, hard_pred, 5)
            # mip_no_vessels_soft = vessels_removal_soft(post, soft_pred, d=2, thresh=0.15, strong=False)
            # mip_no_vessels_soft_strong = vessels_removal_soft(post, soft_pred, d=4, thresh=0.1, strong=False)
            # mip_no_vessels_soft2 = vessels_removal_soft(post, soft_pred, thresh=0.2)
            # mip_no_vessels_soft3 = vessels_removal_soft(post, soft_pred, thresh=0.15)

            # plt.imshow(np.max(hard_pred, 2).transpose(1, 0), cmap="gray")
        # plt.show()
        # plt.imshow(np.max(soft_pred, 2).transpose(1, 0), cmap="gray")
        # plt.show()


            plt.subplot(3, 4, 1+4*j)
            plt.imshow(mip.transpose(1, 0), cmap="gray")
            if j==0:
                plt.title("Original")
                plt.ylabel("Model A")
            elif j == 1:
                plt.ylabel("Model B")
            elif j==2:
                plt.ylabel("Model C")
            plt.xticks([])
            plt.xticks([])
            plt.subplot(3, 4, 2+4*j)
            plt.imshow(vessels_removal_hard(post, hard_pred, 5).transpose(1, 0), cmap="gray")
            if j==0:
                plt.title("Algorithm A")
            plt.xticks([])
            plt.xticks([])
            plt.subplot(3, 4, 3+4*j)
            plt.imshow(vessels_removal_soft2(post, soft_pred, d=2, thresh=0.3, scale=30).transpose(1, 0), cmap="gray")
            if j==0:
                plt.title("Algorithm B")
            plt.xticks([])
            plt.xticks([])
            plt.subplot(3, 4, 4+4*j)
            plt.imshow(vessels_removal_soft2(post, soft_pred, d=4, thresh=0.2, scale=40).transpose(1, 0), cmap="gray")
            if j==0:
                plt.title("Algorithm B Strong")
            plt.xticks([])
            plt.xticks([])

        plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)

        plt.savefig(os.path.join("/cache/marco/results/vessel_removal/duke/", id+".png"))