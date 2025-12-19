from typing import Dict
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os
import pandas as pd
import torch
import inspect
import SimpleITK as sitk


def generate_DukeMRI_Seg_MS_metadata(path_seq_map_file, images_root, sequences, label_root):
    # TODO: rewrite (too many modification result in a inefficient and unreadable code)
    path_seq = path_seq_map(path_seq_map_file)
    slice_order = dcm_pos_orientation(images_root)
    # 1 decreasing (not normal) 0 increasing (normal)

    path_seq = path_seq[path_seq["sequence"].isin(sequences)]

    ids_with_label = os.listdir(label_root)
    path_seq = path_seq[path_seq["classic_path"].str.split("/").str[1].isin(ids_with_label)]

    a = pd.DataFrame([(x, slice_order[x]) for x in slice_order], columns=["classic_path", "slice_ordering"])
    a = pd.merge(a, path_seq)

    data = []
    for x in ids_with_label:
        d = {}
        for sequence in sequences:
            item_seq_ides = a["classic_path"][(a["sequence"] == sequence) & (a["classic_path"].str.contains(x))]
            if len(item_seq_ides) == 1:
                tmp = item_seq_ides.item()
            else:
                tmp = None
            d[sequence] = tmp
        data.append((x, a["slice_ordering"][(a["classic_path"].str.contains(x)) & (a["sequence"]==sequences[0])].item(), d, os.path.join("Segmentation_Masks_NRRD", x, "Segmentation_" + x + "_Dense_and_Vessels.seg.nrrd")))

    data = pd.DataFrame(data, columns=["id", "slice_ordering", "sequence_paths", "label_path"])

    return data


def load3dfrom2dslice(path, flip_slice_dim=False):
    return np.stack([pydicom.dcmread(os.path.join(path, filename)).pixel_array for filename in sorted(os.listdir(path), reverse=flip_slice_dim)])


def load3dfrom2dslice_sitk(path):
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_filenames = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_filenames)

    # Load the image series as a 3D volume
    dicom_image = reader.Execute()

    # Convert the SimpleITK image to a NumPy array
    dicom_array = sitk.GetArrayFromImage(dicom_image)

    return dicom_array


def path_seq_map(mapping_file):
    """ Work with classic filepath (not descriptive)
    return the paths relative to the directory containing the patients (Breast_MRI_001, Breast_MRI_002...)

    :param mapping_file: filepath to Breast-Cancer-MRI-filepath_filename-mapping.csv
    :return: pandas DF containing sequence-path association
    """
    map = pd.read_csv(mapping_file, sep=',')
    map = map[["original_path_and_filename", "classic_path"]]
    map["original_path_and_filename"] = map["original_path_and_filename"].str.split("/").str[-2]
    map = map.rename(columns={'original_path_and_filename': 'sequence'})

    map['classic_path'] = map['classic_path'].str.rsplit("/", n=1).str[0]

    map = map.drop_duplicates(subset="classic_path")

    return map


def dcm_pos_orientation(root_dir):

    result = {}

    patients = os.listdir(os.path.join(root_dir, "Duke-Breast-Cancer-MRI"))
    patients.remove("LICENSE")

    for patient in patients:
        study = os.listdir(os.path.join(root_dir, "Duke-Breast-Cancer-MRI", patient))[0]

        sequences = os.listdir(os.path.join(root_dir, "Duke-Breast-Cancer-MRI", patient, study))
        sequences = [x for  x in sequences if "egmentatio" not in x]

        for sequence in sequences:

            slices = sorted(os.listdir(os.path.join(root_dir, "Duke-Breast-Cancer-MRI", patient, study, sequence)))
            if len(slices) == 1:
                continue

            first_slice = pydicom.dcmread(os.path.join(root_dir, "Duke-Breast-Cancer-MRI", patient, study, sequence, slices[0]))
            z_first = first_slice.ImagePositionPatient[2]
            z_last = pydicom.dcmread(os.path.join(root_dir, "Duke-Breast-Cancer-MRI", patient, study, sequence, slices[-1])).ImagePositionPatient[2]

            if z_first > z_last:
                result[os.path.join("Duke-Breast-Cancer-MRI", patient, study, sequence)] = 1
            else :
                result[os.path.join("Duke-Breast-Cancer-MRI", patient, study, sequence)] = 0

    return result


def model_rgb2gray(model):
    # identify first layer

    first_layer = model
    while len(list(first_layer.children())) > 1:
        first_layer = list(first_layer.children())[0]

    # convert first layer to process grayscale image
    first_layer.in_channels = 1
    first_layer.weight = torch.nn.Parameter(first_layer.weight.sum(1, keepdim=True))


def create_figure(res: Dict[str, list], path: str, verbose: bool = True) -> None:
    """
    Create the figures accuracy, loss, mcc and ROC using the result of an experiment.

    :param res: a dictonary containing the result of the form {"epochs": [], "losses": [], "mcc": []...}
    :type res: Dict[str, list]
    :param path: the directory where save the figures
    :type path: str
    :param verbose: if True print messages during execution
    :type verbose: bool
    """
    epochs = res["epochs"]

    if verbose:
        print("Saving figures...")

    # loss
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"loss")
    ax.plot(epochs, res['losses'], label=f"min={min(res['losses']):.4f} at epoch {np.argmin(res['losses']) + 1}")
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "loss.png"))
    plt.close(fig)

    # train and validation accuracy
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"accuracy")
    ax.plot(epochs, res["train_accuracies"],
            label=f'train (max={max(res["train_accuracies"]):.3f} at epoch {np.argmax(res["train_accuracies"]) + 1})')
    ax.plot(epochs, res["valid_accuracies"],
            label=f'valid (max={max(res["valid_accuracies"]):.3f} at epoch {np.argmax(res["valid_accuracies"]) + 1})')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "accuracy.png"))
    plt.close(fig)

    # mcc
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_title(f"mcc")
    ax.plot(epochs, np.abs(res["mcc"]),
            label=f'mcc (max={max(res["mcc"]):.3f} at epoch {np.argmax(res["mcc"]) + 1})')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(path, "mcc.png"))
    plt.close(fig)

    # roc at epoch with largest auc, only if res["auc"] is not empty (binary case)
    if len(res["auc"]) != 0:
        epoch_best = np.argmax(res["auc"])
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        ax.set_title(f"ROC")
        ax.plot(res["roc"][epoch_best][0], res["roc"][epoch_best][1],
                label=f'AUC={res["auc"][epoch_best]:.3f}')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(path, "roc.png"))
        plt.close(fig)

    if verbose:
        print("Done.")


def has_parameter(cls, param_name):
    init_signature = inspect.signature(cls.__init__)
    return param_name in init_signature.parameters


def monai_dict(path="/data/cantone/datasets/Duke-Breast-Cancer-MRI/other/metadata_complete"):
    labeled_ids = os.listdir("/data/cantone/datasets/Duke-Breast-Cancer-MRI/Segmentation_Masks_NRRD/")
    labeled_ids.remove("Breast_MRI_246")
    labeled_ids.remove("Breast_MRI_435")

    metadata = torch.load(path)

    metadata = metadata[metadata["id"].isin(labeled_ids)]

    monai_data = []

    root = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"

    for i in range(len(metadata)):
        tmp = {}
        tmp["pre"] = os.path.join(root, metadata.iloc[i]["sequence_paths"]["pre"])
        tmp["post_1"] = os.path.join(root, metadata.iloc[i]["sequence_paths"]["post_1"])
        tmp["post_2"] = os.path.join(root, metadata.iloc[i]["sequence_paths"]["post_2"])
        if metadata.iloc[i]["sequence_paths"]["post_3"]:
            tmp["post_3"] = os.path.join(root, metadata.iloc[i]["sequence_paths"]["post_3"])
        else:
            tmp["post_3"] = None
        if metadata.iloc[i]["sequence_paths"]["post_4"]:
            tmp["post_4"] = os.path.join(root, metadata.iloc[i]["sequence_paths"]["post_4"])
        else:
            tmp["post_4"] = None
        tmp["seg"] = os.path.join(root, metadata.iloc[i]["label_path"])
        tmp["breast"] = os.path.join(root, metadata.iloc[i]["label_path"]).replace("Dense_and_Vessels", "Breast")
        monai_data.append(tmp)

    return monai_data


from pathlib import Path
import pandas as pd
import os
from typing import List, Dict


def build_duke_mri_monai_dict(
    mapping_file: str,
    dataset_root: str,
    label_root: str,
    sequences: List[str],
    drop_ids: List[str] | None = None,
) -> List[Dict]:
    """
    Build MONAI-compatible dataset dict for Duke Breast Cancer MRI.

    Parameters
    ----------
    mapping_file : str
        Path to Breast-Cancer-MRI-filepath_filename-mapping.csv
    dataset_root : str
        Root directory of Duke-Breast-Cancer-MRI dataset
    label_root : str
        Root directory containing Segmentation_Masks_NRRD
    sequences : list[str]
        Sequence names (e.g. ["pre", "post_1", "post_2", "post_3", "post_4"])
    drop_ids : list[str], optional
        Patient IDs to exclude

    Returns
    -------
    list[dict]
        MONAI-style dataset list
    """

    dataset_root = Path(dataset_root)
    label_root = Path(label_root)

    # ------------------------------------------------------------------
    # 1. Load and clean mapping file
    # ------------------------------------------------------------------
    df = pd.read_csv(mapping_file, usecols=["original_path_and_filename", "classic_path"])

    df["sequence"] = df["original_path_and_filename"].str.split("/").str[-2]
    df["image_path"] = df["classic_path"].str.rsplit("/", n=1).str[0]
    df["id"] = df["image_path"].str.split("/").str[1]

    df = df[df["sequence"].isin(sequences)]

    # ------------------------------------------------------------------
    # 2. Keep only labeled patients
    # ------------------------------------------------------------------
    labeled_ids = set(os.listdir(label_root))

    if drop_ids:
        labeled_ids -= set(drop_ids)

    df = df[df["id"].isin(labeled_ids)]

    # ------------------------------------------------------------------
    # 3. Build MONAI dict
    # ------------------------------------------------------------------
    monai_data = []

    for pid, group in df.groupby("id"):
        item = {}

        # image sequences
        for seq in sequences:
            row = group[group["sequence"] == seq]

            if len(row) > 0:
                # all slices share the same parent directory
                seq_dir = Path(row.iloc[0]["image_path"]).parent
                item[seq] = str(dataset_root / seq_dir)
            else:
                item[seq] = None

        # labels
        seg_path = dataset_root / "Segmentation_Masks_NRRD" / pid / f"Segmentation_{pid}_Dense_and_Vessels.seg.nrrd"
        item["seg"] = str(seg_path)
        item["breast"] = str(seg_path).replace("Dense_and_Vessels", "Breast")

        monai_data.append(item)

    return monai_data



def generate_monai_AMBL(dataset_path):
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

