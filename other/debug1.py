from utilities import build_duke_mri_monai_dict

monai_data = build_duke_mri_monai_dict(
    mapping_file="/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/Breast-Cancer-MRI-filepath_filename-mapping.csv",
    dataset_root="/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI",
    label_root="/ssd1/cantone/datasets/Duke-Breast-Cancer-MRI/Segmentation_Masks_NRRD",
    sequences=["pre", "post_1", "post_2", "post_3", "post_4"],
    drop_ids=["Breast_MRI_246", "Breast_MRI_435"],
)
