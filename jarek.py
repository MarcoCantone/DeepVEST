import torch
import yaml
import importlib

from monai.transforms import Compose
from monai.data import Dataset
from monai.transforms import Transform, ToTensor

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

def load_config(config_file=None):
    if config_file is None:
        return cfg
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_class_by_path(path):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def create_object_from_dict(dict):
    cls = get_class_by_path(dict["class"])
    for key, value in dict["params"].items():
        if type(value) == type({}):
            dict["params"][key] = create_object_from_dict(value)
    return cls(**dict["params"])

if __name__ == "__main__":

    cfg = load_config("path to cfg file")

    train_transforms = None
    if "TRAIN_TRANSFORM" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["TRAIN_TRANSFORM"])):
            transform_list.append(create_object_from_dict(cfg["TRAIN_TRANSFORM"][i]))
        train_transforms = Compose(transform_list)

    test_transforms = None
    if "TEST_TRANSFORM" in cfg.keys():
        transform_list = []
        for i in range(len(cfg["TEST_TRANSFORM"])):
            transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
        test_transforms = Compose(transform_list)


    # train_data is a list of dictionary that define the path of each element of each sample of the dataset
    # Is a common data structure for work with the Monai Dataset class
    # The following id an example. I have created my own for the duke dataset (the way i have saved the images and labels)
    train_data = [
        {
            "pre_contrast": "path to pre-contrast image",
            "post_contrast": "path to post-contrast image",
            "seg_map": "path to annotation",
        },
        {
            "same for the second sample of the dataset"

        }
        #...
    ]

    train_dataset = Dataset(train_data, train_transforms)
