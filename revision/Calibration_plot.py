import torch
from monai.inferers import sliding_window_inference

from config import *
from sklearn.model_selection import train_test_split
import os
from monai.data import Dataset, DataLoader, decollate_batch, PydicomReader, MetaTensor

from monai.transforms import Compose, AsDiscrete
from monai.metrics import ConfusionMatrixMetric, HausdorffDistanceMetric
from customMetrics import DiceMIP

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt



class MIP_projection():
    def __call__(self, x):
        return x.max(-1).values

def sigmoid_fn(z):
    return 1/(1 + np.exp(-z))



if __name__ == "__main__":
    device = torch.device("cuda:2")

    folders = {
        "/data/cantone/experiments/MRI/paper_experiments/seed34_Unet_wholeVolume_preContrast/":
            "U-Net",
        "/data/cantone/experiments/MRI/paper_experiments/seed34_Unet_preContrast/":
            "U-Net, Sub-vol.",
        "/data/cantone/experiments/MRI/paper_experiments/seed34_Unet/":
            "U-Net, Sub-vol., Dual Seq.",
        "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/":
            "DeepVEST"
    }
    plt.figure(figsize=(5, 5), dpi=300)
    for folder, label in folders.items():
        print(folder)
        # folder = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet/"
        dataset_root = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
        monai_metadata = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative"
        model_path = os.path.join(folder, "best_model.pth")
        save_path = ""
        save = False

        cfg = load_config(os.path.join(folder, "config.yaml"))

        correct_reader = True
        if correct_reader:
            cfg["TEST_TRANSFORM"][0]["params"]["reader"] = {"class": "monai.data.PydicomReader", "params": {}}

        data = torch.load(monai_metadata)

        load_seg_mip = False
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
        else:
            if "seg_mip" not in cfg["TEST_TRANSFORM"][0]["params"]["keys"]:
                cfg["TEST_TRANSFORM"][0]["params"]["keys"] += ["seg_mip"]
                cfg["TEST_TRANSFORM"][1]["params"]["keys"] += ["seg_mip"]
                cfg["TEST_TRANSFORM"][6]["params"]["keys"] += ["seg_mip"]
                cfg["TEST_TRANSFORM"][5]["params"]["keys"].remove("seg_mip")
                cfg["TEST_TRANSFORM"].insert(-2, {"class": "monai.transforms.SqueezeDimd", "params": {"keys": ["seg_mip"], "dim": -1}})

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

        # metric_names = ["sensitivity", "matthews correlation coefficient", "f1_score"]
        # metric = ConfusionMatrixMetric(include_background=False, reduction="none", metric_name=metric_names)

        metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)


        model.eval()

        y_pred_list = []
        y_true_list = []

        for i in range(len(test_dataset)):
            print(i)
            with torch.no_grad():
                # id = test_dataset.data[i]["seg_mip"].split("/")[6]
                sample = test_dataset[i]
                val_inputs, val_labels = (
                    sample["img"][None, ...].to(device),
                    sample["seg"][None, ...].to(device)
                )

                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                # val_outputs.shape = [1, 2, H, W, D]

                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                val_outputs = sigmoid_fn(val_outputs[0][1]-val_outputs[0][0])
                # val_outputs.shape = [H, W, D]

                y_pred_list.append(val_outputs)
                y_true_list.append(val_labels[0].cpu())

        y_pred = np.concatenate([x.flatten() for x in y_pred_list])
        y_true = np.concatenate([x[1].flatten() for x in y_true_list])



        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

        plt.plot(prob_pred, prob_true, marker='o', label=label)


    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives", fontsize=12)
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.title("Calibration plot (reliability diagram)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    # Equal axes and limits
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # Create legend with header
    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(
        handles, labels,
        loc="upper left",
        fontsize=9,
        title_fontsize=11,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig("/home/cantone/Calibration_plot.png")
    plt.savefig("/home/cantone/Calibration_plot.pdf")