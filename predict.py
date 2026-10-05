"""3D vessel segmentation with the DeepVEST segmentation model.

    # Duke test set (15 cases)
    python predict.py --dataset duke

    # AMBL dataset (external, no fine-tuning)
    python predict.py --dataset ambl

For every case the binary vessel segmentation is written to
``<output-dir>/<case_id>_vessels.nii.gz`` (and, with ``--save-probability``, the vessel
probability map to ``<case_id>_vessel_probability.nii.gz``). The images are reoriented to LPS
before inference, and the outputs are saved in that orientation with the corresponding affine.

The full volume is segmented with sliding-window inference (96^3 sub-volumes, 25% overlap);
the vessel label is the argmax of the two output classes, i.e. sigmoid(s_vessel - s_background) > 0.5.
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from config import load_config
from utilities import inference_transforms, input_sequences, load_cases, load_model, sample_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D vessel segmentation with DeepVEST.")
    parser.add_argument("--dataset", type=str, choices=["duke", "ambl"], required=True)
    parser.add_argument("--config", type=str, default="configs/deepvest_segmentation.yaml", help="model configuration")
    parser.add_argument("--weights", type=str, default="weights/deepvest_segmentation.pth", help="model weights")
    parser.add_argument("--datalist", type=str, default=None, help="default: data/<dataset>_datalist_local.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"], help="Duke cases to process")
    parser.add_argument("--split-dir", type=str, default=None, help="default: TRAINING.split_dir of the configuration")
    parser.add_argument("--output-dir", type=str, default=None, help="default: outputs/segmentation/<dataset>")
    parser.add_argument("--save-probability", action="store_true", help="also save the vessel probability map")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    datalist = args.datalist if args.datalist is not None else f"data/{args.dataset}_datalist_local.json"
    split_dir = args.split_dir if args.split_dir is not None else cfg["TRAINING"]["split_dir"]
    output_dir = args.output_dir if args.output_dir is not None else os.path.join("outputs", "segmentation", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    data = load_cases(args.dataset, datalist, args.split, split_dir)
    transform = inference_transforms(args.dataset, input_sequences(cfg))
    model = load_model(cfg, args.weights, device)

    for i, elem in enumerate(data):
        name = sample_name(elem)
        print(f"[{i + 1}/{len(data)}] {name}")
        sample = transform(dict(elem))
        img = sample["img"]

        with torch.no_grad():
            soft = sliding_window_inference(img[None, ...].to(device), [96, 96, 96], 4, model)

        output = (torch.sigmoid(soft[0, 1] - soft[0, 0]) > 0.5).to(torch.uint8)
        affine = img.affine.numpy()

        nib.save(nib.Nifti1Image(output.cpu().numpy(), affine), os.path.join(output_dir, f"{name}_vessels.nii.gz"))
        if args.save_probability:
            probability = torch.sigmoid(soft[0, 1] - soft[0, 0]).cpu().numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(probability, affine), os.path.join(output_dir, f"{name}_vessel_probability.nii.gz"))

    print(f"Saved {len(data)} segmentation(s) to {output_dir}")
