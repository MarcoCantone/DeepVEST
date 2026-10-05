"""Custom image readers."""

import numpy as np
from monai.data.image_reader import NrrdReader


class PatchedNrrdReader(NrrdReader):
    """``NrrdReader`` that also reads 3D Slicer segmentations with a non-spatial axis.

    Some Duke ``.seg.nrrd`` annotations store the segments as separate layers (4D volume,
    ``kinds: list domain domain domain``). The ``space directions`` of the layer axis is ``none``
    (a row of NaN), which makes MONAI's ``NrrdReader`` fail when building the affine. Here the
    non-spatial rows are discarded and the affine is built from the three spatial axes only.
    The voxel data are read unchanged.
    """

    def _get_affine(self, header):
        direction = np.asarray(header["space directions"], dtype=float)
        direction = direction[~np.isnan(direction).any(axis=1)]  # keep the spatial axes only
        direction = direction[:3, :3]
        affine = np.eye(4)
        affine[:3, :3] = direction.T
        affine[:3, 3] = np.asarray(header["space origin"], dtype=float)[:3]
        return affine
