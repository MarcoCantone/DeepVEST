# import cv2
# import os
#
# # --- User-provided paths ---
# image1_path = r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\removal\cropped\AMBL-559_MIP_cropped.png"
# image2_path = r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\removal\cropped\AMBL-559_MIP_no_vessels_cropped.png"
# # Load images
# img1 = cv2.imread(image1_path)
# img2 = cv2.imread(image2_path)
# name1 = image1_path.split("\\")[-1][:-4]
# name2 = image2_path.split("\\")[-1][:-4]
# if img1 is None or img2 is None:
#     print("Error loading images. Check the paths.")
#     exit()
# clone = img1.copy()
# crop_coords = []
# drawing = False
# def draw_square(event, x, y, flags, param):
#     global crop_coords, drawing, clone, img1
#     if event == cv2.EVENT_LBUTTONDOWN:
#         crop_coords = [(x, y)]
#         drawing = True
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         img1 = clone.copy()
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         cv2.rectangle(img1, (x0, y0), (x1, y1), (0, 255, 255), 2)  # Yellow rectangle
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         crop_coords.append((x1, y1))
#         cv2.rectangle(img1, (x0, y0), (x1, y1), (0, 255, 255), 2)
# cv2.namedWindow("Draw square on Image 1")
# cv2.setMouseCallback("Draw square on Image 1", draw_square)
# print("Draw a square on Image 1. Press any key when done.")
# while True:
#     cv2.imshow("Draw square on Image 1", img1)
#     key = cv2.waitKey(1) & 0xFF
#     if key != 255:  # Any key pressed
#         break
# cv2.destroyAllWindows()
# if len(crop_coords) < 2:
#     print("No square drawn. Exiting.")
#     exit()
# # Sort coordinates
# (x0, y0), (x1, y1) = crop_coords
# x_min, x_max = sorted([x0, x1])
# y_min, y_max = sorted([y0, y1])
# save_folder = r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\removal\zoom"
# # Draw and save annotated image1
# annotated_img1 = clone.copy()
# cv2.rectangle(annotated_img1, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
# print(os.path.join(save_folder, name1+"_square.png"))
# cv2.imwrite(os.path.join(save_folder, name1+"_square.png"), annotated_img1)
# # Crop and save both images
# crop_img1 = clone[y_min:y_max, x_min:x_max]
# crop_img2 = img2[y_min:y_max, x_min:x_max]
# cv2.imwrite(os.path.join(save_folder, name1+"_zoom.png"), crop_img1)
# cv2.imwrite(os.path.join(save_folder, name2+"_zoom.png"), crop_img2)
# print("Saved: image1_with_square.png, image1_cropped.png, image2_cropped.png")


# import cv2
# import numpy as np
# import os
#
# # --- Load 4 image paths ---
# image_paths = []
# image_paths.append(r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_mip_cropped.png")
# image_paths.append(r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_mip_model_cropped.png")
# image_paths.append(r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_slice_cropped.png")
# image_paths.append(r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_slice_model_cropped.png")
#
# save_path = r"C:\Users\marco\Desktop\test"
#
# ids = [x.split("\\")[-1][:-4] for x in image_paths]
#
# images = [cv2.imread(p) for p in image_paths]
#
# if any(img is None for img in images):
#     print("One or more images could not be loaded. Check the paths.")
#     exit()
#
# # --- Let user draw square on first image ---
# img1 = images[0]
# clone = img1.copy()
# img1_display = img1.copy()
# crop_coords = []
# drawing = False
#
# def draw_square(event, x, y, flags, param):
#     global crop_coords, drawing, img1_display
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         crop_coords.clear()
#         crop_coords.append((x, y))
#         drawing = True
#
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         img1_display = clone.copy()
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         crop_coords.append((x1, y1))
#         cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
#
# cv2.namedWindow("Draw square on Image 1")
# cv2.setMouseCallback("Draw square on Image 1", draw_square)
#
# print("Draw a square on Image 1. Press any key when done.")
# while True:
#     cv2.imshow("Draw square on Image 1", img1_display)
#     if cv2.waitKey(1) & 0xFF != 255:
#         break
# cv2.destroyAllWindows()
#
# if len(crop_coords) < 2:
#     print("No square drawn. Exiting.")
#     exit()
#
# # --- Coordinates ---
# (x0, y0), (x1, y1) = crop_coords
# x_min, x_max = sorted([x0, x1])
# y_min, y_max = sorted([y0, y1])
# square_size = x_max - x_min
#
# # --- Processing for each image ---
# for idx, img in enumerate(images):
#     original = img.copy()
#
#     # Crop from img1 (not from each image)
#     crop = images[idx][y_min:y_max, x_min:x_max]
#
#     # Upsample to full width of the original image
#     width = original.shape[1]
#     scale_factor = width / square_size
#     new_height = int(square_size * scale_factor)
#     crop_resized = cv2.resize(crop, (width, new_height), interpolation=cv2.INTER_NEAREST)
#
#     # Draw yellow square on original image
#     annotated = original.copy()
#     cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#
#     # Draw yellow border on cropped version
#     cv2.rectangle(crop_resized, (0, 0), (width - 1, new_height - 1), (0, 255, 255), 2)
#
#     # White gap between images
#     gap = 20
#     gap_img = 255 * np.ones((gap, width, 3), dtype=np.uint8)
#
#     # Stack vertically
#     stacked = np.vstack([annotated, gap_img, crop_resized])
#
#     # Draw connecting lines
#     offset_y = original.shape[0] + gap
#     cv2.line(stacked, (x_min, y_max), (0, offset_y), (0, 255, 255), 2)  # BL → TL
#     cv2.line(stacked, (x_max, y_max), (width - 1, offset_y), (0, 255, 255), 2)  # BR → TR
#
#     # Save result
#     out_name = ids[idx]+"_with_zoom.png"
#     cv2.imwrite(os.path.join(save_path, out_name), stacked)
#     print(f"Saved: {out_name}")

# import cv2
# import numpy as np
# import os
#
# # --- User Configuration ---
# merge_on_top = False  # Set to False to put cropped image at the bottom
# color = (0, 165, 255)
#
# # --- Input Paths ---
# image_paths = [
#     r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_mip_cropped.png",
#     r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_mip_model_cropped.png",
#     r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_slice_cropped.png",
#     r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\qualitative_ambl\cropped\AMBL-559_slice_model_cropped.png"
# ]
#
# save_path = r"C:\Users\marco\Desktop\test"
# ids = [os.path.basename(p)[:-4] for p in image_paths]
#
# # --- Load Images ---
# images = [cv2.imread(p) for p in image_paths]
# if any(img is None for img in images):
#     print("One or more images could not be loaded. Check the paths.")
#     exit()
#
# # --- Draw Square on First Image ---
# img1 = images[0]
# clone = img1.copy()
# img1_display = img1.copy()
# crop_coords = []
# drawing = False
#
#
# def draw_square(event, x, y, flags, param):
#     global crop_coords, drawing, img1_display
#     if event == cv2.EVENT_LBUTTONDOWN:
#         crop_coords.clear()
#         crop_coords.append((x, y))
#         drawing = True
#     elif event == cv2.EVENT_MOUSEMOVE and drawing:
#         img1_display = clone.copy()
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         x0, y0 = crop_coords[0]
#         side = min(abs(x - x0), abs(y - y0))
#         x1 = x0 + side if x >= x0 else x0 - side
#         y1 = y0 + side if y >= y0 else y0 - side
#         crop_coords.append((x1, y1))
#         cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
#
#
# cv2.namedWindow("Draw square on Image 1")
# cv2.setMouseCallback("Draw square on Image 1", draw_square)
#
# print("Draw a square on Image 1. Press any key when done.")
# while True:
#     cv2.imshow("Draw square on Image 1", img1_display)
#     if cv2.waitKey(1) & 0xFF != 255:
#         break
# cv2.destroyAllWindows()
#
# if len(crop_coords) < 2:
#     print("No square drawn. Exiting.")
#     exit()
#
# # --- Extract Coordinates ---
# (x0, y0), (x1, y1) = crop_coords
# x_min, x_max = sorted([x0, x1])
# y_min, y_max = sorted([y0, y1])
# square_size = x_max - x_min
#
# # --- Process and Save Images ---
# for idx, img in enumerate(images):
#     original = img.copy()
#     crop = images[idx][y_min:y_max, x_min:x_max]  # Always crop from img1
#
#     width = original.shape[1]
#     scale_factor = width / square_size
#     new_height = int(square_size * scale_factor)
#     crop_resized = cv2.resize(crop, (width, new_height), interpolation=cv2.INTER_NEAREST)
#
#     # Draw yellow rectangle on both original and crop
#     annotated = original.copy()
#     cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.rectangle(crop_resized, (0, 0), (width - 1, new_height - 1), color, 6)
#
#     gap = 20
#     gap_img = 255 * np.ones((gap, width, 3), dtype=np.uint8)
#
#     if merge_on_top:
#         stacked = np.vstack([crop_resized, gap_img, annotated])
#         # Draw connecting lines from bottom crop to original top square
#         offset_y = new_height + gap
#         cv2.line(stacked, (0, new_height), (x_min, y_min + offset_y), color, 2)
#         cv2.line(stacked, (width - 1, new_height), (x_max, y_min + offset_y), color, 2)
#     else:
#         stacked = np.vstack([annotated, gap_img, crop_resized])
#         offset_y = original.shape[0] + gap
#         cv2.line(stacked, (x_min, y_max), (0, offset_y), color, 2)
#         cv2.line(stacked, (x_max, y_max), (width - 1, offset_y), color, 2)
#
#     out_name = ids[idx] + "_with_zoom.png"
#     cv2.imwrite(os.path.join(save_path, out_name), stacked)
#     print(f"Saved: {out_name}")
#



import cv2
import numpy as np
import os

# --- User Configuration ---
merge_on_top = False  # Set to False to put cropped image at the bottom
color = (0, 165, 255)

# --- Input Paths ---
image_paths = [
    r"C:\Users\marco\Downloads\DeepVEST_ Vessel Segmentation and Removal\figure\removal\cropped\Breast_MRI_525_MIP_cropped.png"
]

save_path = r"C:\Users\marco\Desktop\test"
ids = [os.path.basename(p)[:-4] for p in image_paths]

# --- Load Images ---
images = [cv2.imread(p) for p in image_paths]
if any(img is None for img in images):
    print("One or more images could not be loaded. Check the paths.")
    exit()

images_no_vessel = [cv2.imread(p.replace("_cropped", "_no_vessels_cropped")) for p in image_paths]
if any(img is None for img in images):
    print("One or more images could not be loaded. Check the paths.")
    exit()

# --- Draw Square on First Image ---
img1 = images[0]
clone = img1.copy()
img1_display = img1.copy()
crop_coords = []
drawing = False


def draw_square(event, x, y, flags, param):
    global crop_coords, drawing, img1_display
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_coords.clear()
        crop_coords.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img1_display = clone.copy()
        x0, y0 = crop_coords[0]
        side = min(abs(x - x0), abs(y - y0))
        x1 = x0 + side if x >= x0 else x0 - side
        y1 = y0 + side if y >= y0 else y0 - side
        cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = crop_coords[0]
        side = min(abs(x - x0), abs(y - y0))
        x1 = x0 + side if x >= x0 else x0 - side
        y1 = y0 + side if y >= y0 else y0 - side
        crop_coords.append((x1, y1))
        cv2.rectangle(img1_display, (x0, y0), (x1, y1), (0, 255, 255), 2)


cv2.namedWindow("Draw square on Image 1")
cv2.setMouseCallback("Draw square on Image 1", draw_square)

print("Draw a square on Image 1. Press any key when done.")
while True:
    cv2.imshow("Draw square on Image 1", img1_display)
    if cv2.waitKey(1) & 0xFF != 255:
        break
cv2.destroyAllWindows()

if len(crop_coords) < 2:
    print("No square drawn. Exiting.")
    exit()

# --- Extract Coordinates ---
(x0, y0), (x1, y1) = crop_coords
x_min, x_max = sorted([x0, x1])
y_min, y_max = sorted([y0, y1])
square_size = x_max - x_min

# --- Process and Save Images ---
for idx, img in enumerate(images):
    original = img.copy()
    crop = images[idx][y_min:y_max, x_min:x_max]  # Always crop from img1

    width = original.shape[1]
    scale_factor = width / square_size
    new_height = int(square_size * scale_factor)
    crop_resized = cv2.resize(crop, (width, new_height), interpolation=cv2.INTER_NEAREST)

    # Draw yellow rectangle on both original and crop
    annotated = original.copy()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.rectangle(crop_resized, (0, 0), (width - 1, new_height - 1), color, 6)

    gap = 20
    gap_img = 255 * np.ones((gap, width, 3), dtype=np.uint8)

    stacked = np.vstack([annotated, gap_img, crop_resized])
    offset_y = original.shape[0] + gap
    cv2.line(stacked, (x_min, y_max), (0, offset_y), color, 2)
    cv2.line(stacked, (x_max, y_max), (width - 1, offset_y), color, 2)

    out_name = ids[idx] + "_with_zoom.png"
    cv2.imwrite(os.path.join(save_path, out_name), stacked)
    print(f"Saved: {out_name}")

for idx, img in enumerate(images_no_vessel):
    original = img.copy()
    crop = images_no_vessel[idx][y_min:y_max, x_min:x_max]  # Always crop from img1

    width = original.shape[1]
    scale_factor = width / square_size
    new_height = int(square_size * scale_factor)
    crop_resized = cv2.resize(crop, (width, new_height), interpolation=cv2.INTER_NEAREST)

    # Draw yellow rectangle on both original and crop
    annotated = original.copy()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.rectangle(crop_resized, (0, 0), (width - 1, new_height - 1), color, 6)

    gap = 20
    gap_img = 255 * np.ones((gap, width, 3), dtype=np.uint8)


    stacked = np.vstack([crop_resized, gap_img, annotated])
    # Draw connecting lines from bottom crop to original top square
    offset_y = new_height + gap
    cv2.line(stacked, (0, new_height), (x_min, y_min + offset_y), color, 2)
    cv2.line(stacked, (width - 1, new_height), (x_max, y_min + offset_y), color, 2)


    out_name = ids[idx] + "_no_vessels_with_zoom.png"
    cv2.imwrite(os.path.join(save_path, out_name), stacked)
    print(f"Saved: {out_name}")

