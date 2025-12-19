import torch
from monai.inferers import sliding_window_inference

from config import *
from sklearn.model_selection import train_test_split
import os
from monai.transforms import Compose, AsDiscrete, LoadImaged, EnsureChannelFirstd, ConcatItemsd, Orientationd, \
    ScaleIntensityd, DeleteItemsd
from monai.data import Dataset, DataLoader, decollate_batch, PydicomReader
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from math import ceil
from monai.transforms import Compose, AsDiscrete, Lambda
import time
from customMetrics import DiceMIP

from customTransform import SliceOrderingd, separate_vessels_breast


def test(test_loader, model, metric_fn, post_pred, post_label, device=torch.device("cpu"), print_per_epoch=5, roi_size=(96, 96, 96), adjust_metric=False, sliding_window=True):
    model.eval()

    verbose_step = ceil(len(test_loader) / print_per_epoch)

    with torch.no_grad():
        start_time = time.time()
        for step_i, val_data in enumerate(test_loader):
            val_inputs, val_labels = (
                val_data["img"].to(device),
                val_data["seg"].to(device),
            )
            sw_batch_size = 4
            if sliding_window:
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            else:
                val_outputs = model(val_inputs)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # compute metric for current iteration

            if adjust_metric:
                # FIXME: val_data["breast"] can be inverted in z-axis
                num_classes = len(val_outputs[0])
                breast = val_data["breast"].to(device)
                for batch_i in range(len(val_outputs)):
                    for class_i in range(1, num_classes):
                        val_outputs[batch_i][class_i] = torch.where(breast[batch_i][0] == 1, val_outputs[batch_i][class_i], 0)

            metric_fn(y_pred=val_outputs, y=val_labels)

            if step_i % verbose_step == verbose_step - 1:
                print(
                    f"test progress={100. * (step_i + 1) / len(test_loader):.1f}%\t"
                    f"elapsed time={time.time() - start_time:.3f} s")
                start_time = time.time()

        # metric = metric_fn.aggregate().item()
        # reset the status for next validation round
        # metric_fn.reset()

        return metric_fn


if __name__ == "__main__":
    device = torch.device("cuda:2")
    folder = "/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet_3class/"
    cfg = load_config(os.path.join(folder, "config.yaml"))

    # cfg["TEST_TRANSFORM"][0]["params"]["keys"] = ['pre', 'post_2', 'seg', 'breast']
    # cfg["TEST_TRANSFORM"][1]["params"]["keys"] = ['pre', 'post_2', 'seg', 'breast']
    # cfg["TEST_TRANSFORM"][6]["params"]["keys"] = ['pre', 'post_1', 'post_2', 'post_3', 'post_4', 'vessels', 'dense', 'background']
    # cfg["TEST_TRANSFORM"][7]["params"]["keys"] = ['img', 'seg', 'breast']

    correct_reader = True
    if correct_reader:
        cfg["TEST_TRANSFORM"][0]["params"]["reader"] = {"class": "monai.data.PydicomReader", "params": {}}

    load_breast = False
    if load_breast:
        cfg["TEST_TRANSFORM"][0]["params"]["keys"] += ["breast"]
        cfg["TEST_TRANSFORM"][1]["params"]["keys"] += ["breast"]
        cfg["TEST_TRANSFORM"][5]["params"]["keys"].pop(cfg["TEST_TRANSFORM"][5]["params"]["keys"].index("breast"))
        cfg["TEST_TRANSFORM"][6]["params"]["keys"] += ["breast"]

    # data = torch.load(cfg["TRAINING"]["monai_data"], weights_only=True)
    data = torch.load("/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/other/data_monai_relative")

    load_seg_mip = False
    if "dataset_root" in cfg["TRAINING"].keys() and cfg["TRAINING"]["dataset_root"] is not None:
        # root = cfg["TRAINING"]["dataset_root"]
        root = "/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/"
        for elem in data:
            if load_seg_mip:
                id = elem["seg"].split("/")[1]
                mip_relative_path = os.path.join("MIP post annotation", id, "vessels.seg.nrrd")
                if os.path.exists(os.path.join(root, mip_relative_path)):
                    elem["seg_mip"] = mip_relative_path
                else:
                    elem["seg_mip"] = None
            for key in elem:
                if isinstance(elem[key], str):
                    elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

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

    # post_label = Compose([
    #     Lambda(func=lambda x: x.max(dim=-1).values),  # take max over slice dimension
    #     AsDiscrete(to_onehot=2)  # convert to one-hot
    # ])

    test_dataset = Dataset(test_data, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TRAINING"]["test_batch_size"], shuffle=False, num_workers=1)

    model = create_object_from_dict(cfg["MODEL"]).to(device)
    # model.load_state_dict(torch.load(os.path.join(cfg["TRAINING"]["workspace"], "best_model.pth"), weights_only=True, map_location=device))
    model.load_state_dict(torch.load("/data/cantone/experiments/MRI/paper_experiments/seed34_AttentionUnet_3class/best_model.pth", weights_only=True,
                                     map_location=device))

    roi_size = [96, 96, 96]

    #metric = DiceMetric(include_background=False, reduction="none")
    metric_names = ["sensitivity", "matthews correlation coefficient", "f1_score"]
    metric = ConfusionMatrixMetric(include_background=False, reduction="none", metric_name=metric_names)

    #metric = DiceMIP(include_background=False, reduction="none")

    res = test(test_loader, model, metric, post_pred, post_label, device, 10, roi_size, adjust_metric=False, sliding_window=cfg["TRAINING"]["sliding_window"])
    res_agg = res.aggregate()

    with open(os.path.join(folder, "metrics.csv"), "w") as f:
        f.write(", ".join(metric_names) + "\n")
        for i in range(len(test_dataset)):
            for j in range(len(metric_names)-1):
                f.write(str(res_agg[j][i][0].item()) + ", ")
            f.write(str(res_agg[len(metric_names)-1][i][0].item()) + "\n")

    print(res_agg[2].mean())
