# DeepVEST: Deep Learning-based Vessel Segmentation and Erasure in Breast MRI

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Official code and trained models of the paper:

> **DeepVEST: Deep Learning-based Vessel Segmentation and Erasure in Breast MRI for Improved Lesion Assessment.**
> Marco Cantone, Tianyu Zhang, Claudio Marrocco, Luyi Han, Antonio Portaluri, Jarek van Dijk, Carla Sitges Puigivila,
> Miguel Braga, Noemi Schmidt, Alessandro Bria, Ritse Mann.
> *Radiology: Artificial Intelligence*, 2026. [doi:10.1148/ryai.250630](https://doi.org/10.1148/ryai.250630)

DeepVEST segments blood vessels in 3D dynamic contrast-enhanced (DCE) breast MRI and removes them from the
maximum intensity projection (MIP), so that lesions are no longer obscured by vascular structures.

| | |
|---|---|
| Vessel segmentation, Duke test set (n = 15) | DSC **0.611**, MCC **0.619**, sensitivity **0.666** |
| Reader study (5 breast radiologists, 150 assessments) | vessel removal effectiveness 3.82 / 5, artifact severity 0.49 / 5 |

---

## Method

DeepVEST uses two [Attention U-Nets](https://arxiv.org/abs/1804.03999) (MONAI implementation, channels
`[32, 64, 128, 256]`, strides `[2, 2, 2]`), trained on 96³ sub-volumes of the Duke-Breast-Cancer-MRI dataset.

1. **Vessel segmentation model.** Its input is the fat-saturated T1-weighted pre-contrast sequence plus the 2nd
   post-contrast sequence (two channels). The checkpoint is selected by the 3D validation DSC.
2. **Vessel-removal model.** Its input is the 2nd post-contrast sequence only. The checkpoint is selected by
   the validation DSC in **MIP space**, i.e. the MIP of the prediction compared with manual MIP annotations.
3. **Vessel-removal algorithm** (applied to the post-contrast volume before the MIP):
   - vessel probability `p = sigmoid(s_vessel − s_background)` from the removal model;
   - grayscale dilation of `p` with a 3D cross structuring element;
   - voxel-wise suppression `I_out = I · (1 − sigmoid(k · (p − t)))`, with `t = 0.2` and `k = 60`;
   - MIP of the suppressed volume.

All full volumes are processed with sliding-window inference (96³ windows, 25% overlap). Segmentation labels are
obtained with an argmax over the two output classes (background, vessel), without threshold tuning.

## Repository structure

```
├── prepare_data.py        # builds the local datalists (Duke / AMBL)
├── train.py               # training (both models)
├── evaluate.py            # per-case DSC / MCC / sensitivity on the Duke test set
├── bootstrap_ci.py        # mean and 95% bootstrap CI (Table 2)
├── predict.py             # 3D vessel segmentation (Duke test set or AMBL)
├── vessel_removal.py      # vessel-free MIPs (Duke test set or AMBL)
├── config.py              # YAML configuration utilities
├── customTransform.py     # custom MONAI transforms
├── customDatasets.py      # NRRD reader for 3D Slicer segmentations (3D or 4D layered)
├── customMetrics.py       # MIP-space DSC (model selection of the removal model)
├── train_functions.py     # training loop
├── utilities.py           # datalist, split and transform helpers
├── configs/               # configurations of the two DeepVEST models
├── data/duke_datalist.json  # the 98 Duke cases used in the paper (relative paths)
├── splits/                # train/validation/test case ids of the two models
├── annotations/mip/       # manual MIP vessel annotations (Duke)
├── weights/               # trained models
└── results/               # per-case reference metrics of the paper (Table 2)
```

## Installation

Tested with Python 3.10, PyTorch 2.5 and MONAI 1.4.

```bash
git clone https://github.com/MarcoCantone/DeepVEST.git
cd DeepVEST
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version first, following the [official instructions](https://pytorch.org/get-started/locally/).

## Data

### Duke-Breast-Cancer-MRI (training and evaluation)

Download the images and the vessel annotations (`Segmentation_Masks_NRRD`) from the
[TCIA page of the collection](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/).
For the images, use **"Classic Directory Name"** in the NBIA Data Retriever. Arrange both under one folder:

```
<DUKE_ROOT>/
├── Duke-Breast-Cancer-MRI/
│   └── Breast_MRI_XXX/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
└── Segmentation_Masks_NRRD/
    └── Breast_MRI_XXX/Segmentation_Breast_MRI_XXX_Dense_and_Vessels.seg.nrrd
```

**Cases.** 100 cases have vessel annotations. Breast_MRI_246 and Breast_MRI_435 were excluded because of
annotation errors, leaving the 98 cases listed in `data/duke_datalist.json`.

**MIP annotations.** The MIP vessel annotations in `annotations/mip/<case_id>/vessels.seg.nrrd` are only needed
to train the models: they drive the MIP-space model selection on the validation cases. They are not needed for
evaluation or inference.

**Datalist.** Create the local datalist, which contains absolute paths:

```bash
python prepare_data.py duke --duke-root <DUKE_ROOT> --mip-dir annotations/mip
```

### Advanced-MRI-Breast-Lesions (external evaluation)

1. **Images.** Download the [Advanced-MRI-Breast-Lesions](https://doi.org/10.7937/C7X1-YN57) (AMBL)
   collection from TCIA, using **"Descriptive Directory Name"**.
2. **Cases.** The cases that have both a registered multi-phase DCE series ("Registered AX Sen Vibrant
   MultiPhase", with the pre-contrast and the four post-contrast phases) and a lesion ROI series are used.
   These are the 99 cases of the paper.
3. **Datalist:**

```bash
python prepare_data.py ambl --ambl-root <AMBL_ROOT>   # <AMBL_ROOT> contains the AMBL-XXX folders
```

## Trained models

| File | Model | Input | Selection |
|---|---|---|---|
| `weights/deepvest_segmentation.pth` | vessel segmentation | pre-contrast + 2nd post-contrast | best validation DSC (3D) |
| `weights/deepvest_removal.pth` | vessel removal | 2nd post-contrast | best validation DSC (MIP space) |

## Reproducing the results

### Vessel segmentation performance (Table 2, DeepVEST row)

```bash
python evaluate.py --config configs/deepvest_segmentation.yaml \
                   --weights weights/deepvest_segmentation.pth \
                   --output outputs/deepvest_test_metrics.csv
python bootstrap_ci.py outputs/deepvest_test_metrics.csv
```

Expected output (the per-case reference values are in `results/deepvest_test_metrics.csv`):

```
DeepVEST, Duke test set (n = 15), mean (bootstrap 95% CI):
  DSC          0.611 (0.533, 0.670)
  MCC          0.619 (0.544, 0.675)
  Sensitivity  0.666 (0.591, 0.732)
```

- **Small differences:** per-case values can differ slightly (around the fourth decimal) because of
  non-deterministic GPU operations.
- **Paper CIs:** in the paper, the bootstrap CIs of all Table 2 models were computed with one shared random
  stream. `bootstrap_ci.py` skips the draws of the three preceding baseline models by default (`--burn-in 9`)
  to reproduce the published intervals exactly. Use `--burn-in 0` for a standalone bootstrap.

### 3D vessel segmentation (Duke test set and AMBL)

```bash
python predict.py --dataset duke    # 15 Duke test cases
python predict.py --dataset ambl    # AMBL cases, no fine-tuning
```

- **Outputs:** `outputs/segmentation/<dataset>/<case_id>_vessels.nii.gz`. The volumes are in LPS
  orientation, with their affine.
- **Probability maps:** add `--save-probability` to also save the vessel probability maps.

### Vessel removal (vessel-free MIPs)

```bash
python vessel_removal.py --dataset duke
python vessel_removal.py --dataset ambl
```

- **Outputs:** `<case_id>_mip.png` (original MIP of the 2nd post-contrast sequence) and
  `<case_id>_mip_no_vessels.png` (vessel-free MIP), in `outputs/vessel_removal/<dataset>/`.
- **Parameters:** the removal parameters can be changed with `--dilation`, `--threshold` and `--slope`. The
  defaults are the paper's values: 3D cross, t = 0.2, k = 60.

### Training

```bash
python train.py --config configs/deepvest_segmentation.yaml   # -> runs/deepvest_segmentation/best_model.pth
python train.py --config configs/deepvest_removal.yaml        # -> runs/deepvest_removal/best_model_mip.pth
```

- **Hyperparameters:** Adam, learning rate 5e-4, 200 epochs, Dice loss, batch size 4 (2 volumes × 2
  gradient-accumulation steps). Each volume contributes four 96³ sub-volumes, sampled 3:1 around vessel and
  background voxels. Images are reoriented to LPS and scaled to [0, 1].
- **Splits:** the paper splits are fixed by `splits/`. The test set (15 cases) is the same for both models.
  The two models were trained with slightly different train/validation splits (73/10 and 74/9 cases).
- **Options:** `--workspace`, `--datalist`, `--split-dir` and `--device` override the values of the
  configuration file.
- **Memory:** `cache_ratio: 1.0` keeps the pre-processed training volumes in RAM (tens of GB for the
  segmentation model). Set it to `null` if memory is limited.
- **Training time:** 200 epochs took about 4.5 h for the segmentation model and about 7.5 h for the removal
  model (without caching), on a single GPU.

## Using DeepVEST on other data

`predict.py` and `vessel_removal.py` accept any datalist in JSON format whose entries point to DICOM series
folders, one per sequence:

```json
[
  {"id": "patient_01", "pre": "/data/patient_01/pre_contrast", "post_2": "/data/patient_01/post_contrast_2"}
]
```

```bash
python predict.py --dataset duke --datalist my_datalist.json --split all --output-dir outputs/my_data
python vessel_removal.py --dataset duke --datalist my_datalist.json --split all --output-dir outputs/my_data
```

- **`--dataset duke`:** use it for data stored as one DICOM series per sequence. The segmentation model needs
  `pre` and `post_2`; the removal model needs only `post_2`.
- **`--dataset ambl`:** use it for a single multi-phase series (`img`).
- **Preprocessing:** the models were trained on fat-saturated T1-weighted DCE-MRI. Before inference, the images
  are reoriented to LPS and min-max scaled.

## Citation

If you use this code or the trained models, please cite:

```bibtex
@article{cantone2026deepvest,
  title   = {DeepVEST: Deep Learning-based Vessel Segmentation and Erasure in Breast MRI for Improved Lesion Assessment},
  author  = {Cantone, Marco and Zhang, Tianyu and Marrocco, Claudio and Han, Luyi and Portaluri, Antonio and
             van Dijk, Jarek and Sitges Puigivila, Carla and Braga, Miguel and Schmidt, Noemi and Bria, Alessandro and
             Mann, Ritse},
  journal = {Radiology: Artificial Intelligence},
  pages   = {e250630},
  year    = {2026},
  doi     = {10.1148/ryai.250630}
}
```

## License

- **This repository:** code, trained models and annotations are released under the
  [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
  (CC BY-NC 4.0). Commercial use is not permitted.
- **Datasets:** the Duke and AMBL datasets are distributed by TCIA under their own licenses and data usage
  policies.
