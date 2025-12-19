import torch
from scipy.ndimage import grey_dilation
from skimage.morphology import octahedron
import cv2 as cv
from torch import nn
from monai.transforms import Compose
from monai.data import Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split
import cv2 as cv

import os
import numpy as np
import imageio

from config import load_config, create_object_from_dict

save_path = "/cache/marco/results/gif/topo/"

device = torch.device("cuda:0")
roi_size = [96, 96, 96]
sw_batch_size = 8

cfg_path = "/cache/marco/experiments/paper_experiments/seed34_AttentionUnet/config.yaml"
cfg = load_config(cfg_path)

data = torch.load(cfg["TRAINING"]["monai_data"], weights_only=True)

if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
    root = cfg["TRAINING"]["dataset_root"]
    for elem in data:
        for key in elem:
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

_, data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])

if "seg_mip" in cfg["TEST_TRANSFORM"][0]["params"]["keys"]:
    cfg["TEST_TRANSFORM"][0]["params"]["keys"] = ['pre', 'post_2', 'seg']
    cfg["TEST_TRANSFORM"][1]["params"]["keys"] = ['pre', 'post_2', 'seg']
    cfg["TEST_TRANSFORM"][6]["params"]["keys"] = ['img', 'seg']
    cfg["TEST_TRANSFORM"][5]["params"]["keys"] = ['pre', 'post_1', 'post_2', 'post_3', 'post_4', 'breast', 'vessels',
                                                  'seg_mip']
    cfg["TEST_TRANSFORM"].pop(-2)

test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)

dataset = Dataset(data, test_transforms)

weights_path = os.path.join(cfg["TRAINING"]["workspace"], "best_model.pth")
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


def vessels_removal_hard(mri, hard_prediction, dilation_size):
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_pred = cv.dilate(hard_prediction, kernel, iterations=1)
    post_no_vessels = np.where(dilated_pred == 1, 0, mri)
    return np.max(post_no_vessels, axis=2)


def vessels_removal_soft(mri, soft_prediction, d = 0, thresh = 0):
    if d > 1:
        if d%2==0:
            structuring_element = octahedron(d//2)
        else:
            structuring_element = np.ones((d, d, d), np.uint8)
        dilated_image = grey_dilation(soft_prediction, footprint=structuring_element)
    else:
        dilated_image = soft_prediction

    # hard removal
    post_no_vessels = np.where(soft_prediction > thresh, 0, mri)

    # soft removal
    post_no_vessels = post_no_vessels*(1-dilated_image)

    # MIP and return
    return np.max(post_no_vessels, axis=2)

# def update_image(_):
#     global post, soft_pred
#     d = cv.getTrackbarPos("strength", "no_vessels")
#     thresh = (100 - cv.getTrackbarPos("sensitivity", "no_vessels")) / 100
#
#     mip_no_vessels_soft = vessels_removal_soft(post, soft_pred, d=2, thresh=0.1)
#
#     cv.imshow("no_vessels", (mip_no_vessels_soft*255).astype(np.uint8))


for i in range(len(dataset)):
# for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print(i)
    sample = dataset[i]
    img = sample["img"]
    post = img[1].numpy()
    label = sample["seg"][0].numpy()

    img = img.to(device)
    print("inference")
    with torch.no_grad():
        soft_pred = sliding_window_inference(img[None, ...], roi_size, sw_batch_size, model)
        pred = [post_pred(i) for i in decollate_batch(soft_pred)]
        vessels = pred[0][1].cpu().numpy()
        # fgt = pred[0][2].cpu().numpy()

    # kernel = np.ones((3, 3), np.uint8)
    # vessels = cv.dilate(vessels, kernel, iterations=1)
    # fgt = cv.dilate(fgt, kernel, iterations=1)
    #
    # post_bgr = post[..., None].repeat(3, 3)
    # post_bgr[:, :, :, 0] = post_bgr[:, :, :, 0] + 0.25 * vessels
    # post_bgr[:, :, :, 1] = post_bgr[:, :, :, 1] + 0.25 * fgt
    # post_bgr /= post_bgr.max()
    # frames = [(post_bgr[:, :, j]*255).astype("uint8").transpose(1, 0, 2) for j in range(post_bgr.shape[2])]
    #
    # print("saving")
    # imageio.mimsave(os.path.join(save_path, str(i)+".gif"), frames)

    soft_pred = soft_pred[0, 0].detach().cpu().numpy()

    post_bgr = post[..., None].repeat(3, 3)
    post_bgr[:, :, :, 2] = post_bgr[:, :, :, 2] + 0.5 * vessels
    post_bgr /= post_bgr.max()

    post_bgr_label = post[..., None].repeat(3, 3)
    post_bgr_label[:, :, :, 2] = post_bgr_label[:, :, :, 2] + 0.5 * label
    post_bgr_label /= post_bgr_label.max()

    mip = np.max(post, axis=2)
    # pred_mip = np.max(vessels, 2)
    mip_overlay = np.max(post_bgr, 2)
    mip_annotation = np.max(post_bgr_label, 2)

    central_slice_index = post.shape[-1]//2
    central_slice = post[:, :, central_slice_index]
    # central_vessels = vessels[:, :, central_slice_index]
    central_slice_overlay = post_bgr[:, :, central_slice_index]
    central_slice_label = post_bgr_label[:, :, central_slice_index]

    os.mkdir(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i)))
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "mip.png"), mip.transpose(1, 0)*255)
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "mip_annotation.png"), mip_annotation.transpose(1, 0, 2)*255)
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "mip_model.png"), mip_overlay.transpose(1, 0, 2)*255)
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "slice.png"), central_slice.transpose(1, 0)*255)
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "slice_annotation.png"), central_slice_label.transpose(1, 0, 2)*255)
    cv.imwrite(os.path.join("/cache/marco/figure_paper/qualitative_duke/", str(i), "slice_model.png"), central_slice_overlay.transpose(1, 0, 2)*255)

    # plt.imshow(mip, cmap="gray")
    # plt.show()
    # plt.imshow(mip_annotation, cmap="gray")
    # plt.show()
    # plt.imshow(mip_overlay, cmap="gray")
    # plt.show()
    #
    # plt.imshow(central_slice, cmap="gray")
    # plt.show()
    # plt.imshow(central_slice_label, cmap="gray")
    # plt.show()
    # plt.imshow(central_slice_overlay)
    # plt.show()




    # soft_pred_mip = nn.Sigmoid()(soft_pred[0, 1]).max(2).values.cpu().numpy()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(mip.transpose(1, 0), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(vessels_removal_hard(post, vessels, 5).transpose(1, 0), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(vessels_removal_soft(post, soft_pred, d=4, thresh=0.05).transpose(1, 0), cmap="gray")
    plt.show()

    # plt.figure(figsize=(40, 40))
    # plt.subplot(4, 4, 1)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=0, thresh=0.6).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 2)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=0, thresh=0.25).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 3)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=0, thresh=0.15).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 4)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=0, thresh=0.05).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 5)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=2, thresh=0.6).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 6)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=2, thresh=0.25).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 7)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=2, thresh=0.15).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 8)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=2, thresh=0.05).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 9)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=3, thresh=0.6).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 10)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=3, thresh=0.25).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 11)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=3, thresh=0.15).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 12)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=3, thresh=0.05).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 13)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=4, thresh=0.6).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 14)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=4, thresh=0.25).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 15)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=4, thresh=0.15).transpose(1, 0), cmap="gray")
    # plt.subplot(4, 4, 16)
    # plt.imshow(vessels_removal_soft(post, soft_pred, d=4, thresh=0.05).transpose(1, 0), cmap="gray")
    #
    # plt.show()


    # cv.namedWindow("no_vessels")
    #
    # cv.createTrackbar("strength", "no_vessels", 0, 7, update_image)
    # cv.createTrackbar("sensitivity", "no_vessels", 0, 100, update_image)
    #
    # while True:
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         break

    # if not os.path.isdir(os.path.join(save_path, str(i))):
    #     os.mkdir(os.path.join(save_path, str(i)))
    # cv.imwrite(os.path.join(save_path, str(i), "mip.png"), mip.transpose(1, 0)*255)
    # cv.imwrite(os.path.join(save_path, str(i), "no_vessels.png"), mip_no_vessels.transpose(1, 0) * 255)
    # cv.imwrite(os.path.join(save_path, str(i), "soft_pred.png"), soft_pred_mip.transpose(1, 0) * 255)
    # cv.imwrite(os.path.join(save_path, str(i), "hard_pred.png"), pred_mip.transpose(1, 0) * 255)
