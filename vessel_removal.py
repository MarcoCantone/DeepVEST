"""DeepVEST vessel removal: vessel-free maximum intensity projections (MIPs).

    # Duke test set (15 cases)
    python vessel_removal.py --dataset duke

    # AMBL dataset (external, no fine-tuning)
    python vessel_removal.py --dataset ambl

Algorithm (Fig. 1b of the paper), applied to the 2nd post-contrast volume:

1. vessel probability map: p = sigmoid(s_vessel - s_background), where s are the output
   scores of the vessel-removal model (sliding-window inference, 96^3 sub-volumes, 25% overlap);
2. smoothing: grayscale dilation of p with a 3D cross structuring element (``--dilation 2``);
3. voxel-wise suppression: I_out = I * (1 - sigmoid(k * (p_dilated - t))), with t = 0.2 and
   k = 60 (``--threshold`` and ``--slope``);
4. MIP of the suppressed volume along the axial (S) axis.

For every case, ``<case_id>_mip.png`` (original MIP) and ``<case_id>_mip_no_vessels.png``
(vessel-free MIP) are written to the output folder, with the same intensity scale.
"""

import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import grey_dilation
from skimage.morphology import octahedron

from config import load_config
from utilities import inference_transforms, input_sequences, load_cases, load_model, sample_name


def sigmoid_fn(z):
    return 1/(1 + np.exp(-z))


def vessels_removal_soft2(mri, soft_prediction, d = 0, thresh = 0.2, scale=60):
    """Suppress vessels in a 3D volume and return its MIP.

    Args:
        mri: post-contrast volume ``[H, W, D]`` (intensities scaled to [0, 1]).
        soft_prediction: vessel probability map ``[H, W, D]``.
        d: size of the structuring element of the grayscale dilation. Even values use a 3D
            cross (``octahedron(d // 2)``; ``d=2`` gives the 3x3x3 cross used in the paper),
            odd values a ``d x d x d`` cube; ``d <= 1`` disables the dilation.
        thresh: center ``t`` of the suppression sigmoid.
        scale: slope ``k`` of the suppression sigmoid.

    Returns:
        The MIP of the vessel-suppressed volume along the last axis, ``[H, W]``.
    """
    if d > 1:
        if d%2==0:
            structuring_element = octahedron(d//2)
        else:
            structuring_element = np.ones((d, d, d), np.uint8)
        dilated_image = grey_dilation(soft_prediction, footprint=structuring_element)
    else:
        dilated_image = soft_prediction

    # soft removal
    post_no_vessels = mri * (1 - sigmoid_fn((dilated_image - thresh) * scale))

    # MIP and return
    return np.max(post_no_vessels, axis=2)


def vessel_probability(model, img, device, roi_size=(96, 96, 96), sw_batch_size=8):
    """Vessel probability map sigmoid(s_vessel - s_background) of a ``[C, H, W, D]`` image."""
    with torch.no_grad():
        soft_pred = sliding_window_inference(img[None, ...].to(device), roi_size, sw_batch_size, model)
        soft_pred = torch.nn.Sigmoid()((soft_pred[0][1] - soft_pred[0][0]).cpu()).numpy()
    return soft_pred


def save_png(image, path):
    """Save a 2D MIP as a grayscale PNG (anterior at the top, display convention of the paper)."""
    plt.imsave(path, image.transpose(1, 0), cmap="gray", vmin=0.0, vmax=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vessel-free MIPs with DeepVEST.")
    parser.add_argument("--dataset", type=str, choices=["duke", "ambl"], required=True)
    parser.add_argument("--config", type=str, default="configs/deepvest_removal.yaml", help="vessel-removal model configuration")
    parser.add_argument("--weights", type=str, default="weights/deepvest_removal.pth", help="vessel-removal model weights")
    parser.add_argument("--datalist", type=str, default=None, help="default: data/<dataset>_datalist_local.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"], help="Duke cases to process")
    parser.add_argument("--split-dir", type=str, default=None, help="default: TRAINING.split_dir of the configuration")
    parser.add_argument("--output-dir", type=str, default=None, help="default: outputs/vessel_removal/<dataset>")
    parser.add_argument("--dilation", type=int, default=2, help="size d of the structuring element (2 = 3D cross)")
    parser.add_argument("--threshold", type=float, default=0.2, help="sigmoid center t")
    parser.add_argument("--slope", type=float, default=60, help="sigmoid slope k")
    parser.add_argument("--save-probability", action="store_true", help="also save the vessel probability map (.nii.gz)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    datalist = args.datalist if args.datalist is not None else f"data/{args.dataset}_datalist_local.json"
    split_dir = args.split_dir if args.split_dir is not None else cfg["TRAINING"]["split_dir"]
    output_dir = args.output_dir if args.output_dir is not None else os.path.join("outputs", "vessel_removal", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    sequences = input_sequences(cfg)  # ["post_2"] for the vessel-removal model
    data = load_cases(args.dataset, datalist, args.split, split_dir)
    transform = inference_transforms(args.dataset, sequences)
    model = load_model(cfg, args.weights, device)
    channel_post = sequences.index("post_2")

    for i, elem in enumerate(data):
        name = sample_name(elem)
        print(f"[{i + 1}/{len(data)}] {name}")
        sample = transform(dict(elem))
        img = sample["img"]
        post = img[channel_post].numpy()

        soft_pred = vessel_probability(model, img, device)

        mip = np.max(post, axis=2)
        mip_no_vessels = vessels_removal_soft2(post, soft_pred, d=args.dilation, thresh=args.threshold, scale=args.slope)

        save_png(mip, os.path.join(output_dir, f"{name}_mip.png"))
        save_png(mip_no_vessels, os.path.join(output_dir, f"{name}_mip_no_vessels.png"))
        if args.save_probability:
            nib.save(nib.Nifti1Image(soft_pred.astype(np.float32), img.affine.numpy()),
                     os.path.join(output_dir, f"{name}_vessel_probability.nii.gz"))

    print(f"Saved {len(data)} vessel-free MIP(s) to {output_dir}")
