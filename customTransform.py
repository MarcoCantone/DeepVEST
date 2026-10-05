"""Custom MONAI transforms used by DeepVEST."""

import torch
from monai.transforms import Transform, ToTensor


class separate_vessels_dense(Transform):
    """Extract binary vessel / dense-tissue masks from a 3D Slicer ``.seg.nrrd`` segmentation.

    The Duke-Breast-Cancer-MRI annotations (``Segmentation_<id>_Dense_and_Vessels.seg.nrrd``)
    store the "Vessels" and "Dense" (fibroglandular tissue) segments either as label values
    of a single 3D volume or as separate layers of a 4D volume. The segment to use is found
    from the segment names stored in the NRRD header.

    The masks are written to ``sample["vessels"]`` / ``sample["dense"]`` (and optionally
    ``sample["background"]``) and the original segmentation key is removed.

    Args:
        seg_key: key of the loaded segmentation.
        channel: whether the segmentation already has a channel dimension
            (i.e. ``EnsureChannelFirstd`` was applied before this transform).
        vessels: output the vessel mask.
        dense: output the dense-tissue mask.
        compute_backgroud: output the background mask (neither vessels nor dense tissue).
        adjust_affine: key of the image whose affine is copied to the output masks.
            If the image slices are stored in decreasing z order (as indicated by the
            ``lastImagePositionPatient`` metadata provided by MONAI's ``PydicomReader``),
            the affine is flipped along z so that the masks are aligned with the image.
    """

    def __init__(self, seg_key, channel=True, vessels=True, dense=True, compute_backgroud=False, adjust_affine=None):
        self.seg_key = seg_key
        self.n_dim = 4 if channel else 3
        self.vessels = vessels
        self.dense = dense
        self.compute_backgroud = compute_backgroud
        if adjust_affine is not None:
            self.affine_key = adjust_affine
        else:
            self.affine_key = None

    def __call__(self, sample):
        int_converter = ToTensor(dtype=torch.uint8)

        segmentation_map = sample[self.seg_key]

        # Identify the segment corresponding to 'Vessels'
        vessel_segment_value = None
        dense_segment_value = None
        vessel_mask = None
        dense_mask = None

        metadata = segmentation_map.meta
        if len(segmentation_map.shape) == self.n_dim:
            # segments stored as label values of a single volume

            # Loop through the metadata to find the segment with name 'Vessels'
            for key, value in metadata.items():
                if 'Segment' in key and 'Name' in key and value == 'Vessels':
                    # Extract the corresponding label value (SegmentX_LabelValue)
                    segment_index = key.split('_')[0]  # Get the segment index (e.g., 'Segment0')
                    vessel_segment_value = metadata[f"{segment_index}_LabelValue"]
                    break
            if vessel_segment_value is None:
                raise ValueError("No 'Vessels' segment found in the metadata.")
            # Extract the segmentation map for the vessels
            vessel_mask = int_converter(segmentation_map == int(vessel_segment_value))

            # Same for the segment with name 'Dense'
            for key, value in metadata.items():
                if 'Segment' in key and 'Name' in key and value == 'Dense':
                    segment_index = key.split('_')[0]
                    dense_segment_value = metadata[f"{segment_index}_LabelValue"]
                    break
            if dense_segment_value is None:
                raise ValueError("No 'Dense' segment found in the metadata.")
            dense_mask = int_converter(segmentation_map == int(dense_segment_value))

        elif len(segmentation_map.shape) == self.n_dim+1:
            # segments stored as separate layers
            if metadata["Segment0_Name"] == "Vessels":
                vessel_layer = metadata["Segment0_Layer"]
                dense_layer = metadata["Segment1_Layer"]
            else:
                vessel_layer = metadata["Segment1_Layer"]
                dense_layer = metadata["Segment0_Layer"]

            if self.n_dim == 3:
                vessel_mask = int_converter(segmentation_map[int(vessel_layer)] // segmentation_map[int(vessel_layer)].max())
                dense_mask = int_converter(segmentation_map[int(dense_layer)] // segmentation_map[int(dense_layer)].max())
            else:
                vessel_mask = int_converter(segmentation_map[:, int(vessel_layer)] // segmentation_map[:, int(vessel_layer)].max())
                dense_mask = int_converter(segmentation_map[:, int(dense_layer)] // segmentation_map[:, int(dense_layer)].max())

        sample.pop(self.seg_key)

        if self.vessels:
            sample["vessels"] = vessel_mask
            if self.affine_key is not None:
                # use the affine of the image; flip z if the DICOM slices are stored in decreasing z order
                sample["vessels"].meta["affine"] = sample[self.affine_key].meta["affine"].clone()
                if sample[self.affine_key].meta["lastImagePositionPatient"][2] < 0:
                    sample["vessels"].meta["affine"][2, 3] += sample["vessels"].shape[-1] * sample["vessels"].meta["affine"][2, 2]
                    sample["vessels"].meta["affine"][:, 2] *= -1
                sample["vessels"].meta["kinds"] = ['domain', 'domain', 'domain']
                if len(segmentation_map.shape) == self.n_dim+1:
                    sample["vessels"].meta["spatial_shape"] = sample["vessels"].meta["spatial_shape"][1:]

        if self.dense:
            sample["dense"] = dense_mask
            if self.affine_key is not None:
                sample["dense"].meta["affine"] = sample[self.affine_key].meta["affine"].clone()
                if sample[self.affine_key].meta["lastImagePositionPatient"][2] < 0:
                    sample["dense"].meta["affine"][2, 3] += sample["dense"].shape[-1] * sample["dense"].meta["affine"][2, 2]
                    sample["dense"].meta["affine"][:, 2] *= -1
                sample["dense"].meta["kinds"] = ['domain', 'domain', 'domain']
                if len(segmentation_map.shape) == self.n_dim+1:
                    sample["dense"].meta["spatial_shape"] = sample["dense"].meta["spatial_shape"][1:]

        if self.compute_backgroud:
            background_mask = (vessel_mask == 0) & (dense_mask == 0)
            sample["background"] = background_mask

            if self.affine_key is not None:
                sample["background"].meta["affine"] = sample[self.affine_key].meta["affine"].clone()
                if sample[self.affine_key].meta["lastImagePositionPatient"][2] < 0:
                    sample["background"].meta["affine"][2, 3] += sample["background"].shape[-1] * \
                                                                 sample["background"].meta["affine"][2, 2]
                    sample["background"].meta["affine"][:, 2] *= -1

        return sample


class splitDCEd(Transform):
    """Split a DCE-MRI series that stores all contrast phases in one volume.

    In the Advanced-MRI-Breast-Lesions (AMBL) dataset, the "Registered AX Sen Vibrant
    MultiPhase" series contains the pre-contrast and the four post-contrast phases stacked
    along the slice dimension. This transform expects a ``[H, W, D]`` tensor, reshapes it to
    ``[num_sequences, H, W, D // num_sequences]`` and keeps the phases listed in ``seqs``
    as channels (``0`` = pre-contrast, ``2`` = second post-contrast phase).
    """

    def __init__(self, key, num_sequences=5, seqs=[0, 2]):
        self.num_sequences = num_sequences
        self.seqs = seqs
        self.key = key

    def __call__(self, sample):
        img = sample[self.key]
        H, W, D = img.shape
        sample[self.key] = img.view(H, W, self.num_sequences, D//self.num_sequences).permute(2, 0, 1, 3)[self.seqs]
        return sample
