import torch
from torch.nn import Sigmoid
import os
import numpy as np
from torch.nn import Sigmoid
import nrrd

import matplotlib.pyplot as plt

from monai.transforms import LoadImaged, Compose, ScaleIntensityd, Orientationd, AsDiscrete, Transposed
from customTransform import splitDCEd
from monai.data import Dataset, decollate_batch
from monai.inferers import sliding_window_inference

from config import create_object_from_dict, load_config

save_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/inference/AMBL_hard/"

data = torch.load("/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels", weights_only=False)
cfg_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/config.yaml"
cfg = load_config(cfg_path)

root = "/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/Advanced-MRI-Breast-Lesions/"
for elem in data:
    for key in elem:
        if isinstance(elem[key], str):
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

device = torch.device("cuda:2")

weights_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/best_model.pth"
model = create_object_from_dict(cfg["MODEL"])
model = model.to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
model.eval()

transform = Compose([LoadImaged(keys=["img", "seg"]),
                     splitDCEd("img", 5, [0, 2]),
                     Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                     Orientationd(keys=["img", "seg"], axcodes="LPS"),
                     ScaleIntensityd(keys=["img"])])

dataset = Dataset(data)

post_pred = None
if "POST_PRED" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["POST_PRED"])):
        transform_list.append(create_object_from_dict(cfg["POST_PRED"][i]))
    post_pred = Compose(transform_list)


for i in range(len(dataset)):
    print(i)
    # if i<50:
    #     continue
    sample = dataset[i]
    id = sample["img"].split("/")[6]
    print("loading")
    sample = transform(sample)

    print("inference")
    with torch.no_grad():
        soft = sliding_window_inference(sample["img"][None, ...].to(device), [96, 96, 96], 4, model)
        hard = [post_pred(i) for i in decollate_batch(soft)]

    output = (Sigmoid()(soft[0, 1]-soft[0, 0]) > 0.5).int()
    output = output.permute(1, 0, 2)

    header = {
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip'
    }

    nrrd.write(os.path.join(save_path, id+"_prediction.nrrd"), output.cpu().numpy(), header)


    # post = sample["img"][1].numpy()
    # vessels = hard
    #
    # post_bgr = post[..., None].repeat(3, 3)
    # post_bgr[:, :, :, 2] = post_bgr[:, :, :, 2] + 0.5 * vessels
    # post_bgr /= post_bgr.max()
    #
    # mip = np.max(post, axis=2)
    # # pred_mip = np.max(vessels, 2)
    # mip_overlay = np.max(post_bgr, 2)
    #
    # central_slice_index = post.shape[-1]//2
    # central_slice = post[:, :, central_slice_index]
    # # central_vessels = vessels[:, :, central_slice_index]
    # central_slice_overlay = post_bgr[:, :, central_slice_index]
    #
    # os.mkdir(os.path.join(save_path, str(i)))
    # cv.imwrite(os.path.join(save_path, str(i), "mip.png"), mip.transpose(1, 0)*255)
    # cv.imwrite(os.path.join(save_path, str(i), "mip_model.png"), mip_overlay.transpose(1, 0, 2)*255)
    # cv.imwrite(os.path.join(save_path, str(i), "slice.png"), central_slice.transpose(1, 0)*255)
    # cv.imwrite(os.path.join(save_path, str(i), "slice_model.png"), central_slice_overlay.transpose(1, 0, 2)*255)

    # print("saving")
    # nrrd.write(os.path.join(save_path, id+".nrrd"), hard)
    #
    # continue
    # exit()

    # img = sample["img"].numpy()
    # post_mip = np.max(img[1], 2)
    #
    # kernel = np.ones((5, 5), np.uint8)
    # dilated_pred = cv.dilate(hard, kernel, iterations=1)
    # tmp = np.where(dilated_pred == 1, 0, img[1])
    # mip_no_vessels = np.max(tmp, 2)
    #
    # mip_bgr = post_mip[..., None].repeat(3, 2)
    # mip_bgr[:, :, 0] = mip_bgr[:, :, 0] + 0.25 * np.max(hard, 2)
    # mip_bgr /= mip_bgr.max()
    #
    # if not os.path.isdir(os.path.join(save_path, id)):
    #     os.mkdir(os.path.join(save_path, id))
    #
    # cv.imwrite(os.path.join(save_path, id, "post_contrast_mip.png"), (post_mip.transpose(1, 0)*255).astype("uint8"))
    # cv.imwrite(os.path.join(save_path, id, "no_vessels.png"), (mip_no_vessels.transpose(1, 0)*255).astype("uint8"))
    # cv.imwrite(os.path.join(save_path, id, "soft_pred.png"), (Sigmoid()(torch.tensor(np.max(soft[1], 2))).numpy().transpose(1, 0)*255).astype("uint8"))
    # cv.imwrite(os.path.join(save_path, id, "hard_pred.png"), (np.max(hard, 2).transpose(1, 0)*255).astype("uint8"))
    # cv.imwrite(os.path.join(save_path, id, "overlay.png"), (mip_bgr.transpose(1, 0, 2)*255).astype("uint8"))



