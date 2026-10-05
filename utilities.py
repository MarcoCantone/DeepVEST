"""Data and configuration helpers shared by the DeepVEST scripts."""

import copy
import json
import os
import re

import torch
from monai.data import PydicomReader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, DeleteItemsd, Orientationd, \
    ScaleIntensityd

from config import create_object_from_dict
from customTransform import splitDCEd


# ----------------------------------------------------------------------------------------
# Datalists
# ----------------------------------------------------------------------------------------

def load_datalist(path):
    """Load a MONAI datalist (list of dictionaries of file paths).

    Both JSON files (as written by ``prepare_data.py``) and files saved with ``torch.save``
    (the format used during the original experiments) are supported.
    """
    if str(path).endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    return torch.load(path, weights_only=True)


def save_datalist(data, path):
    """Save a MONAI datalist as a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def add_root(data, root):
    """Prepend ``root`` to every path of the datalist (``None`` entries are kept)."""
    for elem in data:
        for key in elem:
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None
    return data


def case_id(path):
    """Return the case identifier contained in a file path.

    Duke cases are named ``Breast_MRI_XXX`` and AMBL cases ``AMBL-XXX``.
    """
    match = re.search(r"Breast_MRI_\d+|AMBL-\d+", str(path))
    if match is None:
        raise ValueError(f"No case id found in path: {path}")
    return match.group(0)


def sample_name(elem):
    """Name used for the output files of a datalist entry.

    The optional ``"id"`` entry is used if present; otherwise the Duke/AMBL case id contained
    in the paths, or, for other data, the name of the folder of the first path.
    """
    if elem.get("id"):
        return elem["id"]
    first_path = next(v for k, v in elem.items() if v is not None)
    try:
        return case_id(first_path)
    except ValueError:
        return os.path.basename(os.path.normpath(first_path))


def read_split(split_dir, name):
    """Read the case ids listed in ``<split_dir>/<name>.txt`` (one id per line)."""
    with open(os.path.join(split_dir, f"{name}.txt"), "r") as f:
        return [line.strip() for line in f if line.strip()]


def select_cases(data, ids, id_key="seg"):
    """Return the datalist entries of the cases in ``ids``, in the same order as ``ids``."""
    by_id = {case_id(elem[id_key]): elem for elem in data}
    missing = [x for x in ids if x not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} case(s) of the split are not in the datalist: {missing}")
    return [by_id[x] for x in ids]


# ----------------------------------------------------------------------------------------
# Transforms from configuration files
# ----------------------------------------------------------------------------------------

def compose_from_config(transform_cfgs):
    """Build a ``monai.transforms.Compose`` from a list of ``{class, params}`` dictionaries."""
    transform_cfgs = copy.deepcopy(transform_cfgs)  # create_object_from_dict modifies its input
    return Compose([create_object_from_dict(t) for t in transform_cfgs])


def remove_key_from_transforms(transform_cfgs, key):
    """Return a copy of a transform list that no longer loads/processes the data key ``key``.

    The key is removed from the ``keys`` of every transform (transforms left without keys
    are dropped) and added to the keys of ``DeleteItemsd``, so that the entry is removed
    from the sample before batching. It is used to evaluate/run the models on cases without
    a MIP annotation (``seg_mip``), which is only needed for model selection during training.
    """
    new_cfgs = []
    for t in copy.deepcopy(transform_cfgs):
        keys = t.get("params", {}).get("keys")
        if keys is not None:
            keys = [keys] if isinstance(keys, str) else list(keys)
            if t["class"].endswith("DeleteItemsd"):
                if key not in keys:
                    keys.append(key)
            elif key in keys:
                keys.remove(key)
                if len(keys) == 0:
                    continue
            t["params"]["keys"] = keys
        new_cfgs.append(t)
    return new_cfgs


def input_sequences(cfg):
    """Return the MRI sequences concatenated as network input (e.g. ``["pre", "post_2"]``)."""
    for t in cfg["TEST_TRANSFORM"]:
        if t["class"].endswith("ConcatItemsd") and t["params"].get("name") == "img":
            return list(t["params"]["keys"])
    raise ValueError("No ConcatItemsd transform producing 'img' found in TEST_TRANSFORM.")


# ----------------------------------------------------------------------------------------
# Inference (images only, no annotation needed)
# ----------------------------------------------------------------------------------------

# position of each contrast phase in the AMBL "Registered AX Sen Vibrant MultiPhase" series
AMBL_PHASES = {"pre": 0, "post_1": 1, "post_2": 2, "post_3": 3, "post_4": 4}


def inference_transforms(dataset, sequences):
    """Pre-processing of the input images for inference.

    The DICOM series of the requested ``sequences`` are loaded, concatenated along the channel
    dimension, reoriented to LPS and min-max scaled to [0, 1]. For Duke this is the image branch
    of the training/test transforms of the configuration files; for AMBL the phases are
    extracted from the multi-phase DCE series.

    Args:
        dataset: ``"duke"`` or ``"ambl"``.
        sequences: network input sequences, e.g. ``["pre", "post_2"]`` (segmentation model)
            or ``["post_2"]`` (vessel-removal model).
    """
    if dataset == "duke":
        return Compose([
            LoadImaged(keys=sequences, reader=PydicomReader()),
            EnsureChannelFirstd(keys=sequences),
            ConcatItemsd(keys=sequences, name="img", dim=0),
            DeleteItemsd(keys=sequences),
            Orientationd(keys=["img"], axcodes="LPS"),
            ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        ])
    elif dataset == "ambl":
        return Compose([
            LoadImaged(keys=["img"], reader=PydicomReader()),
            splitDCEd("img", 5, [AMBL_PHASES[s] for s in sequences]),
            Orientationd(keys=["img"], axcodes="LPS"),
            ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        ])
    raise ValueError(f"Unknown dataset: {dataset}")


def load_cases(dataset, datalist, split="test", split_dir=None):
    """Return the datalist entries to process.

    For Duke, ``split`` selects the cases listed in ``<split_dir>/<split>.txt``
    (``"all"`` = every case of the datalist); for AMBL every case of the datalist is used.
    """
    data = load_datalist(datalist)
    if dataset == "duke" and split != "all":
        data = select_cases(data, read_split(split_dir, split))
    return data


def load_model(cfg, weights, device):
    """Create the network described in ``cfg["MODEL"]`` and load its weights."""
    model = create_object_from_dict(copy.deepcopy(cfg["MODEL"]))
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    return model.to(device).eval()


# ----------------------------------------------------------------------------------------
# Advanced-MRI-Breast-Lesions (AMBL)
# ----------------------------------------------------------------------------------------

def generate_monai_AMBL(dataset_path):
    """Build the AMBL datalist (paths relative to ``dataset_path``).

    ``dataset_path`` is the folder that contains one sub-folder per patient (``AMBL-XXX``),
    downloaded from TCIA with descriptive directory names. A case is included if it has
    exactly one registered DCE series ("Registered AX Sen Vibrant MultiPhase", key ``img``)
    and one lesion annotation series ("-ROI", key ``seg``).
    """
    clients = os.listdir(dataset_path)

    monai_data = []

    for client in clients:
        if not os.path.isdir(os.path.join(dataset_path, client)):
            continue
        study = os.listdir(os.path.join(dataset_path, client))[0]
        sequences = os.listdir(os.path.join(dataset_path, client, study))
        img_sequence = [x for x in sequences if "-Registered AX Sen Vibrant MultiPhase" in x]
        seg_sequence = [x for x in sequences if "-ROI" in x]
        if len(img_sequence) != 1 or len(seg_sequence) != 1:
            print(client)
            continue
        d = {
            "img": os.path.join(client, study, img_sequence[0]),
            "seg":  os.path.join(client, study, seg_sequence[0])
        }
        monai_data.append(d)

    return monai_data
