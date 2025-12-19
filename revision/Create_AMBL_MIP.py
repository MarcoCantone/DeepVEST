import os
import torch
from monai.transforms import Compose, LoadImaged, Transposed
from monai.data import Dataset
from customTransform import splitDCEd
import numpy as np
import nibabel as nib

AMBL = False
Duke = True


if AMBL:

    save_folder = r"C:\Users\marco\Desktop\to annotate\AMBL"
    data = torch.load(r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\data_monai_AMBL_registered_labels", weights_only=False)
    root = r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\Advanced-MRI-Breast-Lesions"
    for elem in data:
        for key in elem:
            if isinstance(elem[key], str):
                elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

    with open(r"C:\Users\marco\Desktop\MRI\AMBL_cases_reader_study.txt", "r") as f:
        good_duke = f.readlines()
    good_duke = [x[:-1] for x in good_duke]

    data_subset = [x for x in data if x["img"].split("\\")[6] in good_duke]

    transform = Compose([LoadImaged(keys=["img"]),
                          splitDCEd("img", 5, [2])])

    dataset = Dataset(data_subset, transform)

    for i in range(len(dataset)):
        volume = dataset[i]["img"]
        id = dataset.data[i]["img"].split("\\")[6]
        print(id)
        mip = volume.max(3).values
        mip_3d = mip.array[0, :, :, None]

        nii_img = nib.Nifti1Image(mip_3d, affine=np.eye(4))  # identity affine
        # Save to file
        nib.save(nii_img, os.path.join(save_folder, id+"_mip.nii"))

if Duke:
    save_folder = r"C:\Users\marco\Desktop\to annotate\Duke\MIPs"
    data = torch.load(r"C:\Users\marco\Desktop\MRI\duke\data_monai_relative", weights_only=False)
    root = r"C:\Users\marco\Desktop\MRI\duke"
    for elem in data:
        for key in elem:
            if isinstance(elem[key], str):
                elem[key] = os.path.join(root, elem[key].replace("/", "\\")) if elem[key] is not None else None

    with open(r"C:\Users\marco\Desktop\test_split.txt", "r") as f:
        good_duke = f.readlines()
    good_duke = [x[:-1] for x in good_duke]

    data_subset = [x for x in data if x["post_2"].split("\\")[7] in good_duke]

    transform = Compose([LoadImaged(keys=["post_2"])])

    dataset = Dataset(data_subset, transform)

    for i in range(len(dataset)):
        volume = dataset[i]["post_2"]
        id = dataset.data[i]["post_2"].split("\\")[7]
        print(id)
        mip = volume.max(2).values
        mip_3d = mip.array[:, :, None]
        mip_affine = volume.meta["affine"].clone()

        # Save the original z-axis vector before modifying
        a2_orig = mip_affine[:3, 2].clone()

        # Change the spacing (scale the z-axis)
        mip_affine[2, 2] *= volume.shape[2]

        # Move the origin to the last slice (k_ref = N - 1)
        k_ref = volume.shape[2] - 1
        mip_affine[:3, 3] = mip_affine[:3, 3] + a2_orig * k_ref
        nii_img = nib.Nifti1Image(mip_3d, affine=mip_affine)  # identity affine
        # Save to file
        nib.save(nii_img, os.path.join(save_folder, id+"_mip.nii"))
