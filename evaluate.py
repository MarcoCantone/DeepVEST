"""Evaluate the DeepVEST segmentation model on the Duke test set (Table 2).

For each of the 15 test cases, the sensitivity, the Matthews correlation coefficient (MCC)
and the F1 score (= Dice similarity coefficient, DSC) of the vessel class are computed between
the predicted 3D segmentation and the reference annotation, and written to a CSV file
(one row per case). Use ``bootstrap_ci.py`` on this file to obtain the mean and the 95% CI.

    python evaluate.py --config configs/deepvest_segmentation.yaml \
                       --weights weights/deepvest_segmentation.pth \
                       --output outputs/deepvest_test_metrics.csv

Full volumes are segmented with sliding-window inference (96^3 sub-volumes, 25% overlap) and
the labels are obtained with an argmax over the two output classes (no threshold tuning).
"""

import argparse
import os
import time
from math import ceil

import pandas as pd
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import ConfusionMatrixMetric

from config import load_config, create_object_from_dict
from utilities import load_datalist, add_root, case_id, read_split, select_cases, compose_from_config, \
    remove_key_from_transforms

METRIC_NAMES = ["sensitivity", "matthews correlation coefficient", "f1_score"]


def test(test_loader, model, metric_fn, post_pred, post_label, device=torch.device("cpu"), print_per_epoch=5, roi_size=(96, 96, 96), sliding_window=True):
    """Run the model on every case of ``test_loader`` and accumulate ``metric_fn``."""
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
            metric_fn(y_pred=val_outputs, y=val_labels)

            if step_i % verbose_step == verbose_step - 1:
                print(
                    f"test progress={100. * (step_i + 1) / len(test_loader):.1f}%\t"
                    f"elapsed time={time.time() - start_time:.3f} s")
                start_time = time.time()

        return metric_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-case evaluation of the DeepVEST segmentation model.")
    parser.add_argument("--config", type=str, default="configs/deepvest_segmentation.yaml", help="model configuration")
    parser.add_argument("--weights", type=str, default="weights/deepvest_segmentation.pth", help="model weights")
    parser.add_argument("--datalist", type=str, default=None, help="Duke datalist (default: TRAINING.monai_data)")
    parser.add_argument("--dataset-root", type=str, default=None, help="root prepended to the datalist paths (default: TRAINING.dataset_root)")
    parser.add_argument("--split-dir", type=str, default=None, help="folder with test.txt (default: TRAINING.split_dir)")
    parser.add_argument("--output", type=str, default="outputs/deepvest_test_metrics.csv", help="output CSV file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)

    datalist = args.datalist if args.datalist is not None else cfg["TRAINING"]["monai_data"]
    root = args.dataset_root if args.dataset_root is not None else cfg["TRAINING"].get("dataset_root")
    split_dir = args.split_dir if args.split_dir is not None else cfg["TRAINING"]["split_dir"]

    data = load_datalist(datalist)
    if root is not None:
        data = add_root(data, root)
    test_data = select_cases(data, read_split(split_dir, "test"))

    # same test transforms used during training, without the MIP annotation (not needed here)
    test_transforms = compose_from_config(remove_key_from_transforms(cfg["TEST_TRANSFORM"], "seg_mip"))
    post_pred = compose_from_config(cfg["POST_PRED"])
    post_label = compose_from_config(cfg["POST_LABEL"])

    test_dataset = Dataset(test_data, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=cfg["TRAINING"]["test_batch_size"], shuffle=False, num_workers=1)

    model = create_object_from_dict(cfg["MODEL"]).to(device)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location=device))

    roi_size = [96, 96, 96]

    # vessel class only (include_background=False), one value per case (reduction="none")
    metric = ConfusionMatrixMetric(include_background=False, reduction="none", metric_name=METRIC_NAMES)

    res = test(test_loader, model, metric, post_pred, post_label, device, 10, roi_size, sliding_window=cfg["TRAINING"]["sliding_window"])
    res_agg = res.aggregate()

    results = pd.DataFrame({"case_id": [case_id(x["seg"]) for x in test_data]})
    for j, name in enumerate(METRIC_NAMES):
        results[name] = [res_agg[j][i][0].item() for i in range(len(test_dataset))]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"\nMean over {len(results)} cases: "
          f"DSC={results['f1_score'].mean():.3f}  "
          f"MCC={results['matthews correlation coefficient'].mean():.3f}  "
          f"sensitivity={results['sensitivity'].mean():.3f}")
    print(f"Saved per-case metrics to {args.output}")
