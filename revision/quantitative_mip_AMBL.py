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

save_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/inference/AMBL_hard/"

data = torch.load("/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels", weights_only=False)
# cfg_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/config.yaml"
cfg_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/config.yaml"
cfg = load_config(cfg_path)

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


# transform = Compose([LoadImaged(keys=["img", "seg", "seg_mip"], reader=PydicomReader()),
#                      splitDCEd("img", 5, [0, 2]),
#                      Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
#                      Orientationd(keys=["img", "seg", "seg_mip"], axcodes="LPS"),
#                      ScaleIntensityd(keys=["img"])])

transform = Compose([LoadImaged(keys=["img", "seg", "seg_mip"], reader=PydicomReader()),
                     splitDCEd("img", 5, [2]),
                     Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                     ScaleIntensityd(keys=["img"])])

data = [elem for elem in data if elem["seg_mip"] is not None]
dataset = Dataset(data)

device = torch.device("cuda:2")
# weights_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/best_model.pth"
weights_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/best_model_mip.pth"
model = create_object_from_dict(cfg["MODEL"])
model = model.to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
model.eval()

post_pred = None
if "POST_PRED" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["POST_PRED"])):
        transform_list.append(create_object_from_dict(cfg["POST_PRED"][i]))
    post_pred = Compose(transform_list)


metric_names = ["sensitivity", "matthews correlation coefficient", "f1_score"]
metric = ConfusionMatrixMetric(include_background=False, reduction="none", metric_name=metric_names)

for i in range(len(dataset)):
    print(i)
    sample = dataset[i]
    sample = transform(sample)

    with torch.no_grad():
        soft = sliding_window_inference(sample["img"][None, ...].to(device), [96, 96, 96], 4, model)
        hard = [post_pred(i) for i in decollate_batch(soft)]

    output = (Sigmoid()(soft[0, 1] - soft[0, 0]) > 0.5).int()
    # output = output.permute(1, 0, 2)

    # mip = sample["img"][1].max(2).values
    # label = sample["seg_mip"][:, :, 0]
    # plt.subplot(1, 3, 1)
    # plt.imshow(mip, cmap="gray")
    # plt.subplot(1, 3, 2)
    # plt.imshow(label, cmap="gray")
    # plt.subplot(1, 3, 3)
    # plt.imshow(output.max(2).values.cpu(), cmap="gray")
    # plt.show()

    metric(y_pred=output.max(2).values.cpu()[None, None, ...], y=sample["seg_mip"][:, :, 0][None, None, ...])

print(metric.aggregate())

print("ciao")

# for sample in dataset:
#     sample = transform(sample)
#     mip = sample["img"][1].max(2).values
#     label = sample["seg_mip"][:, :, 0]
#     plt.subplot(1, 2, 1)
#     plt.imshow(mip, cmap="gray")
#     plt.subplot(1, 2, 2)
#     plt.imshow(label, cmap="gray")
#     plt.show()
