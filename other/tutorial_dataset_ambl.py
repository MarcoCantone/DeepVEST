import torch
import os

from monai.transforms import Compose, LoadImaged, Transposed, Orientationd, ScaleIntensityd, Transform
from monai.data import Dataset

import matplotlib.pyplot as plt

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


# prepare data (list of dict)
# eache sample is a dict comprising different key-value pairs like the image, the segmentation map e the classification label
data = torch.load("/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/data_monai_AMBL_registered_labels")
root = "/ssd1/cantone/datasets/Advanced-MRI-Breast-Lesions/Advanced-MRI-Breast-Lesions/"
for elem in data:
    for key in elem:
        if isinstance(elem[key], str):
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

# create transforms
# LoadImaged: open the image in the specified path (the reader is automatically selected)
# slitDCEd: since the image are saved concatenating more volume along the slice dimension,
#   this function split the volumes in the channel dimension e return the volumes specified in the seqs attribute
# TransposeD: rearrange the dimension of segmentation mao in [channel, H, W, D]
# OrientationD: set the orientation of the input
# ScaleIntensityd: MinMax normalization
transform = Compose([LoadImaged(keys=["img", "seg"]),
                     splitDCEd("img", 5, [0, 2]),
                     Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                     Orientationd(keys=["img", "seg"], axcodes="LPS"),
                     ScaleIntensityd(keys=["img"])])

# create dataset object
dataset = Dataset(data, transform)

# load a sample
sample = dataset[0]
image = sample["img"]
segmentation_map = sample["seg"]
label = sample["label"]

# the tensor dimensions are:
# image.shape = [sequences, H, W, D]
# segmentation_map.shape = [lesion_i, H, W, D]

# Show the MIP projection of the post2 sequence (channel 1)
plt.imshow(image[1].max(2).values, cmap="gray")
plt.show()
# show the projected segmentation map of each lesion (some map are missing)
for i in range(segmentation_map.shape[0]):
    plt.imshow(segmentation_map[i].max(2).values, cmap="gray")
    plt.show()

# print the labels (0 normal - 1 malignant). When the segmentation map is missing the corresponding label value is None
print(f"labels = {label}")