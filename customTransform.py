import numpy as np
import torch
from monai.transforms import Transform, ToTensor, MapTransform
from monai.data import MetaTensor
from persistent_homology import compute_betti

 #update

class nrrd_to_vesselMap:

    def __call__(self, x):
        if len(x[0].shape) == 3:
            value = x[1]["Segment0_LabelValue"] if x[1]["Segment0_Name"] == "Vessels" else x[1]["Segment1_LabelValue"]
            x = x[0].transpose(2, 1, 0)
            return np.where(x == int(value), 1, 0)
        elif len(x[0].shape) == 4:
            layer = x[1]["Segment0_Layer"] if x[1]["Segment0_Name"] == "Vessels" else x[1]["Segment1_Layer"]
            return x[0][int(layer)].transpose(2, 1, 0) // x[0][int(layer)].max()


class ExtractVesselSegmentationd(Transform):

    def __init__(self, seg_key):
        self.seg_key = seg_key

    def __call__(self, sample):
        segmentation_map = sample[self.seg_key]

        # Identify the segment corresponding to 'Vessels'
        vessel_segment_value = None

        metadata = segmentation_map.meta
        if len(segmentation_map.shape) == 4:

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
            vessel_mask = (segmentation_map == int(vessel_segment_value)).astype(np.uint8)

        elif len(segmentation_map.shape) == 5:
            layer = metadata["Segment0_Layer"] if metadata["Segment0_Name"] == "Vessels" else metadata["Segment1_Layer"]
            vessel_mask = (segmentation_map[:, int(layer)] // segmentation_map[:, int(layer)].max()).astype(np.uint8)

        sample[self.seg_key] = vessel_mask

        return sample


class separate_vessels_breast(Transform):

    # test upload 1124

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

            for key, value in metadata.items():
                if 'Segment' in key and 'Name' in key and value == 'Dense':
                    # Extract the corresponding label value (SegmentX_LabelValue)
                    segment_index = key.split('_')[0]  # Get the segment index (e.g., 'Segment0')
                    dense_segment_value = metadata[f"{segment_index}_LabelValue"]
                    break
            if dense_segment_value is None:
                raise ValueError("No 'Vessels' segment found in the metadata.")
            # Extract the segmentation map for the vessels
            dense_mask = int_converter(segmentation_map == int(dense_segment_value))

        elif len(segmentation_map.shape) == self.n_dim+1:
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

        # FIXME: In this way the tensor metadata are wrong (like dimension 4) (fixed just for our experiments)
        # TODO: reformat the following if

        if self.vessels:
            sample["vessels"] = vessel_mask
            if self.affine_key is not None:
                # if i not want to use sliceOrdering i have to change the 3 row of affine like it was swapped
                sample["vessels"].meta["affine"] = sample[self.affine_key].meta["affine"]
                sample["vessels"].meta["kinds"] = ['domain', 'domain', 'domain']
                if len(segmentation_map.shape) == self.n_dim+1:
                    sample["vessels"].meta["spatial_shape"] = sample["vessels"].meta["spatial_shape"][1:]

        if self.dense:
            sample["dense"] = dense_mask
            if self.affine_key is not None:
                sample["dense"].meta["affine"] = sample[self.affine_key].meta["affine"]
                sample["dense"].meta["kinds"] = ['domain', 'domain', 'domain']
                if len(segmentation_map.shape) == self.n_dim+1:
                    sample["dense"].meta["spatial_shape"] = sample["dense"].meta["spatial_shape"][1:]

        if self.compute_backgroud:
            background_mask = (vessel_mask == 0) & (dense_mask == 0)
            sample["background"] = background_mask
            if self.affine_key is not None:
                sample["background"].meta["affine"] = sample[self.affine_key].meta["affine"]

        return sample


class separate_vessels_dense(Transform):

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

            for key, value in metadata.items():
                if 'Segment' in key and 'Name' in key and value == 'Dense':
                    # Extract the corresponding label value (SegmentX_LabelValue)
                    segment_index = key.split('_')[0]  # Get the segment index (e.g., 'Segment0')
                    dense_segment_value = metadata[f"{segment_index}_LabelValue"]
                    break
            if dense_segment_value is None:
                raise ValueError("No 'Vessels' segment found in the metadata.")
            # Extract the segmentation map for the vessels
            dense_mask = int_converter(segmentation_map == int(dense_segment_value))

        elif len(segmentation_map.shape) == self.n_dim+1:
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

        # FIXME: In this way the tensor metadata are wrong (like dimension 4) (fixed just for our experiments)
        # TODO: reformat the following if

        if self.vessels:
            sample["vessels"] = vessel_mask
            if self.affine_key is not None:
                # if i not want to use sliceOrdering i have to change the 3 row () not really!) of affine like it was swapped
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


class ExtractVesselSegmentationd_new(Transform):

    def __init__(self, seg_key):
        self.seg_key = seg_key

    def __call__(self, sample):
        segmentation_map = sample[self.seg_key]

        # Identify the segment corresponding to 'Vessels'
        vessel_segment_value = None

        metadata = segmentation_map.meta
        if len(segmentation_map.shape) == 3:

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
            sample[self.seg_key] = (segmentation_map == int(vessel_segment_value))

        elif len(segmentation_map.shape) == 4:
            layer = metadata["Segment0_Layer"] if metadata["Segment0_Name"] == "Vessels" else metadata["Segment1_Layer"]
            sample[self.seg_key] = (segmentation_map[int(layer)] // segmentation_map[int(layer)].max())

        return sample


class Select_sequence(Transform):
    def __init__(self, seq_key):
        self.seq_key = seq_key

    def __call__(self, sample):
        sample["img"] = sample[self.seq_key]
        sample.pop(self.seq_key)
        return sample


class Multi_sequence(Transform):
    def __init__(self, keys, stacked_sequences_name, maintain_keys=[]):
        self.keys = keys
        self.name = stacked_sequences_name
        self.maintain_keys = maintain_keys

    def __call__(self, sample):
        sample[self.name] = torch.stack([sample[x] for x in self.keys])
        for key in self.keys:
            if key in self.maintain_keys:
                continue
            sample.pop(key)
        return sample


class topo_labeld(Transform):

    def __init__(self, seg_map_key):
        self.seg_map = seg_map_key

    def __call__(self, sample):

        if type(sample)==type([]):
            for x in sample:
                x[self.seg_map] = (x[self.seg_map], compute_betti(x[self.seg_map][0])[:-1])
        else:
            sample[self.seg_map] = (sample[self.seg_map], compute_betti(sample[self.seg_map][0])[:-1])

        return sample


class SliceOrderingd(Transform):

    def __init__(self, img_key, keys=None):
        self.img_key = img_key
        self.keys = keys

    def __call__(self, sample):

        img = sample[self.img_key]

        meta = img.meta

        if meta['lastImagePositionPatient'][2] < 0:
            if self.keys is not None:
                for key in self.keys:
                    sample[key] = sample[key].flip(dims=[-1])
                else:
                    sample[self.img_key] = img.flip(dims=[-1])

        return sample


class MIP(Transform):
    def __init__(self, keepdim=False):
        self.keepdim = keepdim

    def __call__(self, volume):
        return volume.max(-1, keepdim=self.keepdim).values


class randomSlices(Transform):
    def __init__(self, keys, n):
        self.keys = keys
        self.n = n

    def __call__(self, sample):
        tot_slices = sample[self.keys[0]].shape[-1]

        slice_list = np.random.choice(range(0, tot_slices), size=self.n, replace=False)

        for key in self.keys:
            sample[key] = sample[key][:, :, :, slice_list]


class splitDCEd(Transform):
    def __init__(self, key, num_sequences=5, seqs=[0, 2]):
    # this transform expect a [H, W, D] tensor where the pre- and post-contrast sequences are stored in D
        self.num_sequences = num_sequences
        self.seqs = seqs
        self.key = key
    def __call__(self, sample):
        img = sample[self.key]
        H, W, D = img.shape
        sample[self.key] =  img.view(H, W, self.num_sequences, D//self.num_sequences).permute(2, 0, 1, 3)[self.seqs]
        return sample


class addTopoLabeld(Transform):
    def __init__(self, keys, y):
        self.keys = keys
        self.y = torch.tensor(y)

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = [sample[key], self.y]


        return sample


class adjust_orientationd(Transform):
    def __init__(self, keys, reference_key):
        self.keys = keys
        self.reference_key = reference_key
    def __call__(self, sample):
        for key in self.keys:
            sample[key].meta["affine"] = torch.matmul(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.double), sample[self.reference_key].meta["affine"])

        return sample

class invertZaxis(Transform):
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key].flip(dims=[-1])

        return sample

class ClipByPercentile(MapTransform):
    """
    Clip the image intensity values based on lower and upper percentiles.
    """
    def __init__(self, keys, lower_percentile=0.1, upper_percentile=99.9):
        super().__init__(keys)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key].astype(np.float32)

            # Compute the percentiles
            lower = np.percentile(img, self.lower_percentile)
            upper = np.percentile(img, self.upper_percentile)

            # Clip the image
            img = np.clip(img, lower, upper)

            d[key] = img
        return d