import numpy as np
import os
import matplotlib.pyplot as plt

import torch

from monai.data import Dataset, decollate_batch, PydicomReader
from monai.transforms import Compose, LoadImaged, Orientationd, ScaleIntensityd, ConcatItemsd, Transposed
from monai.inferers import sliding_window_inference

from sklearn.model_selection import train_test_split

from config import load_config, create_object_from_dict
from customTransform import splitDCEd

from skimage.morphology import octahedron
from scipy.ndimage import grey_dilation


import nibabel as nib

def sigmoid_fn(z):
    return 1/(1 + np.exp(-z))

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


device = torch.device("cuda:2")
roi_size = [96, 96, 96]
sw_batch_size = 8
save = False

cfg_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/config.yaml"
model_name = "best_model_mip.pth"
channel = 1
channel_post = 0
sigmoid = True

duke_monai_data = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative"
dataset_root_duke = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
weights_path = os.path.join("/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/", model_name)

cfg = load_config(cfg_path)

correct_reader = True
if correct_reader:
    cfg["TEST_TRANSFORM"][0]["params"]["reader"] = {"class": "monai.data.PydicomReader", "params": {}}

dataset_type = "duke"
if dataset_type == "duke":
    data = torch.load(duke_monai_data, weights_only=True)

    if "dataset_root" in cfg["TRAINING"].keys() and dataset_root_duke is not None:
        root = dataset_root_duke
        for elem in data:
            for key in elem:
                elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

    if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
        _, data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"],
                                   random_state=cfg["TRAINING"]["train_test_split_seed"])
    else:
        _, data = train_test_split(data, test_size=cfg["TRAINING"]["valid_size"],
                               random_state=cfg["TRAINING"]["train_test_split_seed"])
elif dataset_type == "ambl":
    data = torch.load("/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels",
                      weights_only=False)
    root = "/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/Advanced-MRI-Breast-Lesions/"
    for elem in data:
        for key in elem:
            if isinstance(elem[key], str):
                elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

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

test_type = "cfg"
if test_type == "cfg":
    cfg_path_test_transform = "/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/config_vessels_removal.yaml"
    cfg = load_config(cfg_path_test_transform)
    correct_reader = True
    if correct_reader:
        cfg["TEST_TRANSFORM"][0]["params"]["reader"] = {"class": "monai.data.PydicomReader", "params": {}}
    test_transforms = None
    if "TEST_TRANSFORM" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["TEST_TRANSFORM"])):
            transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
        test_transforms = Compose(transform_list)
elif test_type == "custom":
    test_transforms = Compose([LoadImaged(keys=["img", "seg"], reader=PydicomReader()),
                               splitDCEd("img", 5, [2]),
                               Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                               Orientationd(keys=["img", "seg"], axcodes="LPS"),
                               ScaleIntensityd(keys=["img"])])


dataset = Dataset(data)

# for sample_id in range(len(dataset)):
for sample_id in range(1):
    print(sample_id)
    sample = dataset[sample_id]
    if dataset_type == "duke":
        id = sample["post_2"].split("/")[6]
    else:
        id = sample["img"].split("/")[6]
    sample = test_transforms(sample)
    img = sample["img"]
    post = img[channel_post].numpy()

    print(id)
    exit()

    img = img.to(device)
    print("inference")
    with torch.no_grad():
        soft_pred = sliding_window_inference(img[None, ...], roi_size, sw_batch_size, model)
        hard_pred = [post_pred(i) for i in decollate_batch(soft_pred)]
        if channel == 1:
            soft_pred[0][channel] = soft_pred[0][1] - soft_pred[0][0]
        if sigmoid:
            soft_pred = torch.nn.Sigmoid()(soft_pred[0][channel].cpu()).numpy()
        else:
            soft_pred = soft_pred[0][channel].cpu().numpy()
        hard_pred = hard_pred[0][channel].cpu().numpy()

    mip = np.max(post, axis=2)
    pred_mip_hard = np.max(hard_pred, 2)
    pred_mip_soft = np.max(soft_pred, 2)

    no_vessels = vessels_removal_soft2(post, soft_pred, d=2, thresh=0.5, scale=30)

    if True:
        thresholds = [0.1, 0.2, 0.4, 0.8]
        scales = [15, 60, 120]

        plt.figure(figsize=(7.5*len(scales), 4*len(thresholds)))

        for t_i in range(len(thresholds)):
            for k_i in range(len(scales)):
                no_vessels = vessels_removal_soft2(
                    post, soft_pred, d=2,
                    thresh=thresholds[t_i], scale=scales[k_i]
                )

                plt.subplot(len(thresholds), len(scales), t_i * len(scales) + k_i + 1)
                plt.imshow(no_vessels.transpose(1, 0)[:no_vessels.shape[0]//2, :], cmap="gray")

                if t_i == 0:
                    plt.title(f'k={scales[k_i]}', fontsize=28)
                if k_i == 0:
                    plt.ylabel(f't={thresholds[t_i]}', fontsize=28)

                # hide only ticks and borders, not labels
                ax = plt.gca()
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.tight_layout()
        #plt.show()
        plt.savefig("/home/cantone/removal_parameters.png")
        plt.savefig("/home/cantone/removal_parameters.pdf")

    # lesions = mip.copy()
    # lesions = lesions[..., None].repeat(3, 2)
    #
    # for i in range(len(sample["label"])):
    #     if sample["label"][i] == 1:
    #         lesions[:, :, 0] += 0.5 * np.max(sample["seg"][i], 2)
    #     elif sample["label"][i] == 0:
    #         lesions[:, :, 1] += 0.5 * np.max(sample["seg"][i], 2)
    #
    # lesions /= lesions.max()

    if not save:
        plt.imshow(mip.transpose(1, 0), cmap="gray")
        plt.show()
        plt.imshow(no_vessels.transpose(1, 0), cmap="gray")
        plt.show()
    else:
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(mip.transpose(1, 0), cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(no_vessels.transpose(1, 0), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(lesions.transpose(1, 0, 2))
        plt.savefig(f"/cache/marco/results/vessel_removal/AMBL/png_with_lesions/{id}.png")

        plt.clf()

        # affine = sample["img"].meta["affine"].numpy()
        # affine[2, 2] = affine[2, 2] * sample["img"].shape[-1]
        #
        # mip_3d_np = mip[..., None]
        # nifti_img = nib.Nifti1Image(mip_3d_np, affine)
        # nib.save(nifti_img, f"/cache/marco/results/vessel_removal/AMBL/nifti/{id}_MIP.nii")
        #
        # mip_3d_np = no_vessels[..., None]
        # nifti_img = nib.Nifti1Image(mip_3d_np, affine)
        # nib.save(nifti_img, f"/cache/marco/results/vessel_removal/AMBL/nifti/{id}_MIP_no_vessels.nii")