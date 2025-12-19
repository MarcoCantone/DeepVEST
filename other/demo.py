import os

import cv2
import numpy as np
import torch

from monai.data import Dataset, decollate_batch
from monai.transforms import Compose, AsDiscrete
from config import load_config, create_object_from_dict
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split


cfg_path = "C:\\Users\\marco\\Desktop\\MRI\\8-unet\\config.yaml"
data = torch.load("C:\\Users\\marco\\Desktop\\MRI\\duke\\data_monai_relative", weights_only=False)
root = "C:/Users/marco/Desktop/MRI/duke/"
weights_path = "C:\\Users\\marco\\Desktop\\MRI\\8-unet\\best_model.pth"

cfg = load_config(cfg_path)

for elem in data:
    for key in elem:
        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None
_, data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"], random_state=cfg["TRAINING"]["train_test_split_seed"])
# root_linux = "/data/cantone/datasets/Duke-Breast-Cancer-MRI/"
# root_windows = "C:/Users/marco/Desktop/MRI/duke/"
# downloaded_ids = os.listdir("C:/Users/marco/Desktop/MRI/duke/Duke-Breast-Cancer-MRI")
# data = [{"pre": x["pre"].replace(root_linux, root_windows), "post_1": x["post_1"].replace(root_linux, root_windows), "seg": x["seg"].replace(root_linux, root_windows)} for x in data if x["seg"].split("/")[6] in downloaded_ids]

# create transform
test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)

dataset = Dataset(data, test_transforms)

# create model
model = create_object_from_dict(cfg["MODEL"])
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device("cpu")))
model.eval()

roi_size = [x for x in cfg["TRAIN_TRANSFORM"] if x["class"] == 'monai.transforms.RandCropByPosNegLabeld'][0]["params"]["spatial_size"]
sw_batch_size = 4

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])

for i in [0, 1, 4, 5]:
    # 0,1
    sample = dataset[i]
    print(i)

    img = sample["img"]
    pre = sample["img"][0].numpy()
    post = sample["img"][1].numpy()
    original_label = sample["seg"][0].numpy()
    with torch.no_grad():
        pred = sliding_window_inference(img[None, ...], roi_size, sw_batch_size, model)
        pred = [post_pred(i) for i in decollate_batch(pred)]
        pred = pred[0][1].numpy()

    # pre, post, original_label = sample["pre"][0].numpy(), sample["post_1"][0].numpy(), sample["seg"][0]

    kernel = np.ones((5, 5), np.uint8)
    dilated_label = cv2.dilate(original_label, kernel, iterations=1)
    dilated_pred = cv2.dilate(pred, kernel, iterations=1)

    original_label = original_label / original_label.max()

    # Parameters to control the current slice and mode (slice or MIP)
    current_slice = 0
    mode = 'slice'  # Can be 'slice' or 'mip'
    vessel_removal = False
    use_label = True

    # Callback function for trackbar to update the current slice
    def update_slice(val):
        global current_slice
        current_slice = val
        update_image()

    # Button callback to switch between slice view and MIP
    def switch_mode(event, x, y, flags, param):
        global mode
        global vessel_removal
        global use_label

        if event == cv2.EVENT_LBUTTONDBLCLK:
            # Toggle between 'slice' and 'mip'
            mode = 'mip' if mode == 'slice' else 'slice'
            print(f'Mode: {mode}\n'
                  f'vessel removal: {vessel_removal}\n'
                  f'use_label: {use_label}\n')
            update_image()

        if event == cv2.EVENT_RBUTTONDOWN:
            vessel_removal = not vessel_removal
            print(f'Mode: {mode}\n'
                  f'vessel removal: {vessel_removal}\n'
                  f'use_label: {use_label}\n')
            update_image()

        if event == cv2.EVENT_LBUTTONDOWN:
            use_label = not use_label
            print(f'Mode: {mode}\n'
                  f'vessel removal: {vessel_removal}\n'
                  f'use_label: {use_label}\n')
            update_image()


    # Function to display the current slice or MIP projection
    def update_image():
        if mode == 'slice':
            # Display the current slice
            img_post = post[:, :, current_slice]
            img_pre = pre[:, :, current_slice]
            ann = original_label[:, :, current_slice]
            show_pred = pred[:, :, current_slice]
        elif mode == 'mip':
            # Display the Maximum Intensity Projection (MIP)
            if vessel_removal:
                if use_label:
                    img_post = np.where(dilated_label == 1, 0, post)
                    img_post = np.max(img_post, axis=2)
                    img_pre = np.where(original_label == 1, 0, pre)
                    img_pre = np.max(img_pre, axis=2)
                else:
                    img_post = np.where(dilated_pred == 1, 0, post)
                    img_post = np.max(img_post, axis=2)
                    img_pre = np.where(pred == 1, 0, pre)
                    img_pre = np.max(img_pre, axis=2)
            else:
                img_post = np.max(post, axis=2)  # MIP along the slices axis
                img_pre = np.max(pre, axis=2)

            ann = np.max(original_label, axis=2)
            show_pred = np.max(pred, axis=2)


        cv2.imshow("Post contrast", img_post.transpose(1, 0))
        cv2.imshow("Pre contrast", img_pre.transpose(1, 0))
        cv2.imshow("Annotation", ann.transpose(1, 0))
        cv2.imshow("Model prediction", show_pred.transpose(1, 0))

    # Create the OpenCV window
    cv2.namedWindow("Pre contrast")
    cv2.namedWindow("Post contrast")
    cv2.namedWindow("Annotation")
    cv2.namedWindow("Model prediction")

    # Create a trackbar to scroll through MRI slices
    cv2.createTrackbar("Slice", "Post contrast", 0, post.shape[2] - 1, update_slice)

    # Set mouse callback to switch between slice and MIP mode
    cv2.setMouseCallback("Post contrast", switch_mode)

    # Initialize by showing the first slice
    update_image()

    # Wait for the user to press 'q' to exit the program
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close windows
    cv2.destroyAllWindows()