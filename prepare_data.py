"""Create the local datalists used by the DeepVEST scripts.

    # Duke-Breast-Cancer-MRI (training / evaluation)
    python prepare_data.py duke --duke-root /path/to/Duke --mip-dir annotations/mip

    # Advanced-MRI-Breast-Lesions (external inference)
    python prepare_data.py ambl --ambl-root /path/to/Advanced-MRI-Breast-Lesions

Duke: ``data/duke_datalist.json`` lists the 98 Duke cases used in the paper (paths relative to
the dataset root: DICOM series ``pre``, ``post_1``-``post_4`` and the annotations ``seg`` /
``breast``). This script prepends the dataset root, adds the MIP annotation of each case
(``seg_mip``, ``None`` if not available) and writes ``data/duke_datalist_local.json``.

AMBL: the cases with a registered multi-phase DCE series and a lesion ROI series are
collected from the dataset folder and written to ``data/ambl_datalist_local.json``.

See README.md for the expected folder structure of the datasets.
"""

import argparse
import glob
import os

from utilities import load_datalist, save_datalist, add_root, case_id, generate_monai_AMBL


def find_mip_annotation(mip_dir, cid):
    """Return the MIP annotation of case ``cid`` (``<mip_dir>/<cid>/*.nrrd``), or ``None``."""
    files = sorted(glob.glob(os.path.join(mip_dir, cid, "*.nrrd")))
    if len(files) == 0:
        return None
    vessel_files = [f for f in files if "vessel" in os.path.basename(f).lower()]
    return os.path.abspath(vessel_files[0] if vessel_files else files[0])


def report_missing(data, keys):
    """Print the entries of ``keys`` that do not exist on disk."""
    missing = [(case_id(next(v for v in elem.values() if v is not None)), key, elem[key])
               for elem in data for key in keys if elem.get(key) is not None and not os.path.exists(elem[key])]
    for cid, key, path in missing[:10]:
        print(f"  [missing] {cid} {key}: {path}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    return len(missing)


def prepare_duke(args):
    data = load_datalist(args.reference)
    data = add_root(data, os.path.abspath(args.duke_root))

    n_mip = 0
    for elem in data:
        elem["seg_mip"] = find_mip_annotation(args.mip_dir, case_id(elem["seg"])) if args.mip_dir else None
        n_mip += elem["seg_mip"] is not None

    n_missing = report_missing(data, ["pre", "post_2", "seg"])
    save_datalist(data, args.out)
    print(f"{len(data)} Duke cases ({n_mip} with MIP annotation, {n_missing} missing files) -> {args.out}")


def prepare_ambl(args):
    root = os.path.abspath(args.ambl_root)
    data = generate_monai_AMBL(root)
    data = add_root(data, root)

    n_missing = report_missing(data, ["img"])
    save_datalist(data, args.out)
    print(f"{len(data)} AMBL cases ({n_missing} missing files) -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the local datalists.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    duke = subparsers.add_parser("duke", help="Duke-Breast-Cancer-MRI")
    duke.add_argument("--duke-root", type=str, required=True,
                      help="folder containing Duke-Breast-Cancer-MRI/ (DICOM) and Segmentation_Masks_NRRD/")
    duke.add_argument("--mip-dir", type=str, default="annotations/mip",
                      help="folder with the MIP annotations (<case_id>/vessels.seg.nrrd)")
    duke.add_argument("--reference", type=str, default="data/duke_datalist.json", help="datalist with relative paths")
    duke.add_argument("--out", type=str, default="data/duke_datalist_local.json")

    ambl = subparsers.add_parser("ambl", help="Advanced-MRI-Breast-Lesions")
    ambl.add_argument("--ambl-root", type=str, required=True,
                      help="folder containing one sub-folder per patient (AMBL-XXX)")
    ambl.add_argument("--out", type=str, default="data/ambl_datalist_local.json")

    args = parser.parse_args()
    if args.dataset == "duke":
        prepare_duke(args)
    else:
        prepare_ambl(args)
