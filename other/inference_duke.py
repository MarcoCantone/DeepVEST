from monai.transforms import Compose
from monai.data import Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split

import os
import torch
from torch.nn import Sigmoid
import nrrd
import numpy as np
import nibabel as nib

from config import create_object_from_dict, load_config

save_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/inference/hard/"

data = torch.load("/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative", weights_only=False)
cfg_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/config.yaml"
cfg = load_config(cfg_path)

root = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
for elem in data:
    for key in elem:
        if isinstance(elem[key], str):
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
    data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])
    factor = (len(data)+len(test_data))/len(data)
else:
    factor = 1

data = test_data

device = torch.device("cuda:2")

weights_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/best_model.pth"
model = create_object_from_dict(cfg["MODEL"])
model = model.to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
model.eval()

if "seg_mip" in cfg["TEST_TRANSFORM"][0]["params"]["keys"]:
    cfg["TEST_TRANSFORM"][0]["params"]["keys"] = ['pre', 'post_2', 'seg']
    cfg["TEST_TRANSFORM"][1]["params"]["keys"] = ['pre', 'post_2', 'seg']
    cfg["TEST_TRANSFORM"][6]["params"]["keys"] = ['img', 'seg']
    cfg["TEST_TRANSFORM"][5]["params"]["keys"] = ['pre', 'post_1', 'post_2', 'post_3', 'post_4', 'breast', 'vessels', 'seg_mip']
    cfg["TEST_TRANSFORM"].pop(-2)

transform = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    transform = Compose(transform_list)


dataset = Dataset(data)

post_pred = None
if "POST_PRED" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["POST_PRED"])):
        transform_list.append(create_object_from_dict(cfg["POST_PRED"][i]))
    post_pred = Compose(transform_list)

for i in range(len(dataset)):
    print(i)
    if i<10:
        continue
    if i==7 or i==10:
        continue
    sample = dataset[i]
    id = sample["pre"].split("/")[6]
    sample = transform(sample)

    with torch.no_grad():
        soft = sliding_window_inference(sample["img"][None, ...].to(device), [96, 96, 96], 4, model)
        hard = [post_pred(i) for i in decollate_batch(soft)]

    output = (Sigmoid()(soft[0, 1]-soft[0, 0]) > 0.5).int()
    output = output.permute(1, 0, 2)

    # affine = sample["img"].affine
    #
    # space_directions = [
    #     list(affine[0:3, 0]),
    #     list(affine[0:3, 1]),
    #     list(affine[0:3, 2])
    # ]
    #
    # space_origin = list(affine[0:3, 3])

    header = {
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip'
    }

    nrrd.write(os.path.join(save_path, id+"_prediction.nrrd"), output.cpu().numpy(), header)
