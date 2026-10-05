# MIP vessel annotations (Duke-Breast-Cancer-MRI)

Manual vessel annotations drawn on the maximum intensity projection (MIP) of the 2nd post-contrast
sequence, for 48 cases of the Duke dataset, one folder per case:

```
annotations/mip/<case_id>/vessels.seg.nrrd
```

## How they are used

During training, they are used to compute the validation DSC in MIP space (`customMetrics.DiceMIP`). This
metric selects the checkpoint of the vessel-removal model (`best_model_mip.pth`). The annotations are not
needed to evaluate the models or to run inference.

`prepare_data.py duke --mip-dir annotations/mip` adds them to the datalist (key `seg_mip`).

## Cases required for training

Training needs an annotation for every validation case. The validation cases of the vessel-removal model are a
subset of those of the segmentation model:

- Breast_MRI_021
- Breast_MRI_054
- Breast_MRI_124
- Breast_MRI_280
- Breast_MRI_286
- Breast_MRI_339
- Breast_MRI_503
- Breast_MRI_528
- Breast_MRI_888
- Breast_MRI_902
