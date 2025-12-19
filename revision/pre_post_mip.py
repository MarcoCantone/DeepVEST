import torch
from monai.inferers import sliding_window_inference
import numpy as np

from config import *
from sklearn.model_selection import train_test_split
import os
from monai.data import Dataset, DataLoader, decollate_batch, PydicomReader, MetaTensor

from monai.transforms import Compose, AsDiscrete
from monai.metrics import ConfusionMatrixMetric, HausdorffDistanceMetric
from customMetrics import DiceMIP

import matplotlib.pyplot as plt

import nrrd

class MIP_projection():
    def __call__(self, x):
        return x.max(-1).values



if __name__ == "__main__":
    device = torch.device("cuda:3")
    dataset_root = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
    monai_metadata = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative"
    # folder = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/"
    # model_path = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/best_model.pth"
    folder = "/data/cantone/experiments/MRI/paper_experiments/seed34_post2_selectionMIP/"
    model_path = os.path.join(folder, "best_model_mip.pth")
    save_path = "/data/cantone/experiments/MRI/labels_and_outputs/"
    save = False

    cfg = load_config(os.path.join(folder, "config.yaml"))

    correct_reader = True
    if correct_reader:
        cfg["TEST_TRANSFORM"][0]["params"]["reader"] = {"class": "monai.data.PydicomReader", "params": {}}

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

    if not load_seg_mip:
        if "seg_mip" in cfg["TEST_TRANSFORM"][0]["params"]["keys"]:
            cfg["TEST_TRANSFORM"][0]["params"]["keys"] = ['pre', 'post_2', 'seg']
            cfg["TEST_TRANSFORM"][1]["params"]["keys"] = ['pre', 'post_2', 'seg']
            cfg["TEST_TRANSFORM"][6]["params"]["keys"] = ['img', 'seg']
            cfg["TEST_TRANSFORM"][5]["params"]["keys"] = ['pre', 'post_1', 'post_2', 'post_3', 'post_4', 'breast',
                                                          'vessels', 'seg_mip']
            cfg["TEST_TRANSFORM"].pop(-2)

    if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
        train_data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"],
                                                 random_state=cfg["TRAINING"]["train_test_split_seed"])
    else:
        train_data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["valid_size"],
                                             random_state=cfg["TRAINING"]["train_test_split_seed"])

    test_transforms = None
    if "TEST_TRANSFORM" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["TEST_TRANSFORM"])):
            transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
        test_transforms = Compose(transform_list)


    post_pred = None
    if "POST_PRED" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["POST_PRED"])):
            transform_list.append(create_object_from_dict(cfg["POST_PRED"][i]))
        post_pred = Compose(transform_list)

    post_label = None
    if "POST_LABEL" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["POST_LABEL"])):
            transform_list.append(create_object_from_dict(cfg["POST_LABEL"][i]))
        post_label = Compose(transform_list)

    mip_projection = Compose([
        MIP_projection(),  # take max over slice dimension
        AsDiscrete(to_onehot=2)  # convert to one-hot
    ])

    test_dataset = Dataset(test_data, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TRAINING"]["test_batch_size"], shuffle=False, num_workers=1)

    model = create_object_from_dict(cfg["MODEL"]).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True,
                                     map_location=device))

    roi_size = [96, 96, 96]
    sw_batch_size = 4

    # metric = DiceMIP(include_background=False, reduction="none")

    metric_names = ["sensitivity", "matthews correlation coefficient", "f1_score"]
    metric = ConfusionMatrixMetric(include_background=False, reduction="none", metric_name=metric_names)

    # metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)


    model.eval()

    FP = []
    FFP_TP = []
    IDS = []

    for i in range(len(test_dataset)):
    # for i in range(1):
        print(i)
        with torch.no_grad():
            id = test_dataset.data[i]["seg_mip"].split("/")[6]
            sample = test_dataset[i]
            val_inputs, val_labels, val_mip = (
                sample["img"][None, ...].to(device),
                sample["seg"][None, ...].to(device),
                sample["seg_mip"][None, ...].to(device)
            )

            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            val_outputs_3D = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_3D = [post_label(i) for i in decollate_batch(val_labels)]
            val_labels_2D = [mip_projection(i) for i in decollate_batch(val_labels)]
            val_outputs_2D = [MIP_projection()(i) for i in val_outputs_3D]

            model_output = val_outputs_2D[0][1]
            pre_annotation = val_labels_2D[0][1]
            post_annotation = val_mip[0][0]

            model_output = (model_output > 0.5).astype(np.uint8)
            pre_annotation = (pre_annotation > 0.5).astype(np.uint8)
            post_annotation = (post_annotation > 0.5).astype(np.uint8)

            fp_mask = (model_output == 1) & (pre_annotation == 0)
            tp_in_post_mask = fp_mask & (post_annotation == 1)

            print(fp_mask.sum(), tp_in_post_mask.sum())

            FP.append(fp_mask.sum())
            FFP_TP.append(tp_in_post_mask.sum())
            IDS.append(id)

            h, w = model_output.shape

            plt.figure(figsize=(9, 6), dpi=300)
            plt.subplot(2, 3, 1)
            plt.imshow(model_output.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("Projected model output")
            plt.axis("off")
            plt.subplot(2, 3, 2)
            plt.imshow(pre_annotation.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("Projected pre-contrast label")
            plt.axis("off")
            plt.subplot(2, 3, 3)
            plt.imshow(post_annotation.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("Post-contrast label")
            plt.axis("off")
            plt.subplot(2, 3, 4)
            plt.imshow(val_inputs[0, 0].max(2).values.cpu().transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("MIP")
            plt.axis("off")
            plt.subplot(2, 3, 5)
            plt.imshow(fp_mask.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("FP (pre-contrast)")
            plt.axis("off")
            plt.subplot(2, 3, 6)
            plt.imshow(tp_in_post_mask.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            plt.title("FP that are TP (post-contrast)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join("/data/cantone/experiments/MRI/labels_and_outputs/FP_visual_analysis/", id+".png"))



            # plt.figure(figsize=(7.5, 5), dpi=300)
            # plt.subplot(2, 2, 1)
            # plt.imshow(val_inputs[0, 0].max(2).values.cpu().transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            # plt.title("MIP")
            # plt.axis("off")
            # plt.subplot(2, 2, 2)
            # plt.imshow(model_output.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            # plt.title("Projected model output")
            # plt.axis("off")
            # plt.subplot(2, 2, 3)
            # plt.imshow(pre_annotation.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            # plt.title("Projected pre-contrast label")
            # plt.axis("off")
            # plt.subplot(2, 2, 4)
            # plt.imshow(post_annotation.transpose(1, 0)[int(0.05 * h):int(0.80 * h)], cmap="gray")
            # plt.title("Post-contrast label")
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(os.path.join("/data/cantone/experiments/MRI/labels_and_outputs/output_annotation_comparison_paper/", id+".png"))

            print("ciao")
            # union = torch.logical_or(val_mip[0][0], val_labels_2D[0][1])[None, None, ...]
            # union = MetaTensor(union.to(torch.uint8), meta=val_mip.meta)

            if save:
                model_output_projected = val_outputs_2D[0][1]
                # label_pre_projected = val_labels_2D[0][1]
                # label_post = val_mip[0][0]

                header = {
                    'type': 'float',  # data type
                    'dimension': 2,  # 2D image
                    'sizes': model_output_projected.shape,
                    'encoding': 'raw',  # or 'gzip'
                }

                # nrrd.write(os.path.join(save_path, "label_mip_pre_projected", id+".nrrd", ),
                #            label_pre_projected.cpu().numpy().transpose(1, 0), header)
                #
                # nrrd.write(os.path.join(save_path, "label_mip_post", id + ".nrrd", ),
                #            label_post.cpu().numpy().transpose(1, 0), header)

                nrrd.write(os.path.join(save_path, "modelForSegmentation_output_projected", id + ".nrrd", ),
                           model_output_projected.cpu().numpy().transpose(1, 0), header)

            metric(val_outputs_2D, val_mip)

    for i in range(len(FP)):
        print(FP[i], FFP_TP[i], FFP_TP[i] / FP[i], IDS[i])

    print(f'percentage={np.sum(FFP_TP) / np.sum(FP)}')

    print("done")

    # print(f'Sensitivity={metric.aggregate()[0].mean()}')
    # print(f'MCC={metric.aggregate()[1].mean()}')
    # print(f'DSC={metric.aggregate()[2].mean()}')

    # for i in range(len(test_dataset)):
    #     with torch.no_grad():
    #         id = test_dataset.data[i]["seg_mip"].split("/")[6]
    #         sample = test_dataset[i]
    #         val_inputs, val_labels, val_mip = (
    #             sample["img"][None, ...].to(device),
    #             sample["seg"][None, ...].to(device),
    #             sample["seg_mip"][None, ...].to(device)
    #         )
    #
    #         val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
    #
    #         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
    #         # val_labels_3D = [post_label(i) for i in decollate_batch(val_labels)]
    #         val_labels_2D = [mip_projection(i) for i in decollate_batch(val_labels)]
    #
    #         plt.figure(figsize=(12, 12))
    #
    #         # model output
    #         plt.subplot(2, 2, 1)
    #         plt.imshow(val_outputs[0][1].cpu().max(-1).values.transpose(1, 0), cmap="gray")
    #         plt.axis("off")
    #
    #         plt.subplot(2, 2, 2)
    #         plt.imshow(val_inputs[0, 1].max(-1).values.cpu().transpose(1, 0), cmap="gray")
    #         plt.axis("off")
    #         # label pre projected
    #         plt.subplot(2, 2, 3)
    #         plt.imshow(val_labels_2D[0][1].cpu().transpose(1, 0), cmap="gray")
    #         plt.title(f'DSC={metric(val_outputs, val_labels_2D).item():.3f}')
    #         plt.axis("off")
    #         # label post annotated
    #         plt.subplot(2, 2, 4)
    #         plt.imshow(val_mip[0, 0].cpu().transpose(1, 0), cmap="gray")
    #         plt.title(f'DSC={metric(val_outputs, val_mip).item():.3f}')
    #         plt.axis("off")
    #
    #         if save:
    #             plt.savefig(os.path.join(save_path, f"{id}.png"))
    #         else:
    #             plt.show()
