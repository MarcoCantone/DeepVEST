import torch
import os
import numpy as np

from monai.transforms import LoadImaged, Compose, ScaleIntensityd, Orientationd, AsDiscrete, Transposed
from monai.inferers import sliding_window_inference
from customTransform import splitDCEd
from monai.data import Dataset
from tumor_vessels_intersection import tumor_surface_intersection, vessels_presence_around_tumor
import matplotlib.pyplot as plt

from config import load_config, create_object_from_dict

import cv2 as cv


def update_image(val):
    cv.imshow("Overlay", post_bgr[:, :, val])
    cv.imshow("Original", original[:, :, val])

cfg_path = r"C:\Users\marco\Desktop\MRI\8-unet\config.yaml"
weights_path = r"C:\Users\marco\Desktop\MRI\8-unet\best_model.pth"
data = torch.load(r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\data_monai_AMBL_registered_labels", weights_only=False)
root = r"C:\Users\marco\Desktop\Advanced-MRI-Breast-Lesions\Advanced-MRI-Breast-Lesions"
for elem in data:
    for key in elem:
        if isinstance(elem[key], str):
            elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

transform = Compose([LoadImaged(keys=["img", "seg"]),
                      splitDCEd("img", 5, [0, 2]),
                      Transposed(keys=["seg"], indices=[3, 0, 1, 2]),
                      Orientationd(keys=["img", "seg"], axcodes="LPS"),
                      ScaleIntensityd(keys=["img"])])

dataset = Dataset(data, transform)

cfg = load_config(cfg_path)
device = torch.device("cpu")
model = create_object_from_dict(cfg["MODEL"])
model = model.to(device)
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
model.eval()

i=5
sample = dataset[i]

with torch.no_grad():
    soft = sliding_window_inference(sample["img"][None, ...].to(device), [96, 96, 96], 4, model)[0].cpu().numpy()
    vessels = AsDiscrete(argmax=True)(soft)[0].numpy().astype(np.uint8)

lesions = sample["seg"].max(0).values.numpy()
intersection = tumor_surface_intersection(lesions, vessels, 13, 11)

print(intersection.sum(), sample["label"])

original = sample["img"][1].detach().cpu().numpy()

post_bgr = sample["img"][1][..., None].detach().cpu().numpy().repeat(3, 3)
post_bgr[:, :, :, 0] = post_bgr[:, :, :, 0] + 0.25 * vessels - 0.25 * intersection
post_bgr[:, :, :, 1] = post_bgr[:, :, :, 1] + 0.25 * intersection
post_bgr[:, :, :, 2] = post_bgr[:, :, :, 2] + 0.25 * lesions - 0.25 * intersection
post_bgr /= post_bgr.max()

plt.imshow(np.max(post_bgr, 2))
plt.show()
plt.imshow(np.max(original, 2), cmap="gray")
plt.show()

cv.namedWindow("Overlay", cv.WINDOW_NORMAL)
cv.namedWindow("Original", cv.WINDOW_NORMAL)

cv.createTrackbar("Slice", "Overlay", 0, post_bgr.shape[2]-1, update_image)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()