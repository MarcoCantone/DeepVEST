import torch
import os

data = torch.load("/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai_relative_old")
path_seg_mip = "/cache/marco/datasets/Duke-Breast-Cancer-MRI/MIP_post_annotation/"

annotated_mip_ids = os.listdir(path_seg_mip)

for elem in data:
    id = elem["pre"].split("/")[1]
    if id in annotated_mip_ids:
        elem["seg_mip"] = os.path.join("MIP_post_annotation", id, "vessels.seg.nrrd")
    else:
        elem["seg_mip"] = None

torch.save(data, "/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai_relative")

