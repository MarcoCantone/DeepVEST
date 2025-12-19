import torch
from torch.nn import Sigmoid
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, ConcatItemsd, EnsureChannelFirstd, AsDiscreted, ScaleIntensityd, \
    Orientationd, DeleteItemsd, RandSpatialCropSamples, Resize, RandSpatialCropSamplesd, LoadImage, RandAffined, \
    RandCropByPosNegLabeld, RandFlipd, AsDiscrete, ScaleIntensity, SqueezeDim, SqueezeDimd
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from math import ceil
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config import load_config, create_object_from_dict
from persistent_homology import PH, topo_loss_fn
from itertools import product


def postproc_ph(pred, max_features, connected_component, lr=1, iters=500, e=None, verbose=False, print_per_epoch=10):
    # FIXME substitute with assert
    if pred.max() > 1.0 or pred.min() < 0.0:
        print(f'Pred should be between 0 and 1')

    running_tensor = torch.tensor(pred).requires_grad_(True)
    ph = PH(max_features=max_features)
    loss_fn = topo_loss_fn

    verbose_step = ceil(iters / print_per_epoch)

    losses = []

    for i in range(iters):
        intervals = ph(running_tensor)
        intervals = intervals[:1]
        loss = loss_fn(intervals, [connected_component])
        losses.append(loss.item())
        loss.backward()

        if verbose:
            if i % verbose_step == verbose_step - 1:
                print(f'Iteration {i}')
                plt.imshow(running_tensor.detach().cpu(), cmap="gray")
                plt.show()

        running_tensor = torch.clamp(running_tensor - lr * running_tensor.grad, min=0.0, max=1.0)
        running_tensor = running_tensor.detach().clone().requires_grad_(True)

    return running_tensor.detach(), losses


cfg = load_config("/cache/marco/experiments/vessel_segmentation/99990-split/42/config.yaml")

data = torch.load("/cache/marco/datasets/Duke-Breast-Cancer-MRI/data_monai_relative", weights_only=True)
root = "/cache/marco/datasets/Duke-Breast-Cancer-MRI/"
for elem in data:
    for key in elem:
        elem[key] = os.path.join(root, elem[key]) if elem[key] is not None else None

if "test_size" in cfg["TRAINING"].keys() and cfg["TRAINING"]["test_size"] is not None:
    data, test_data = train_test_split(data, test_size=cfg["TRAINING"]["test_size"],
                                       random_state=cfg["TRAINING"]["train_test_split_seed"])
    factor = (len(data) + len(test_data)) / len(data)
else:
    factor = 1

train_data, valid_data = train_test_split(data, test_size=factor * cfg["TRAINING"]["valid_size"],
                                          random_state=cfg["TRAINING"]["train_test_split_seed"])

test_transforms = None
if "TEST_TRANSFORM" in cfg.keys():
    transform_list = []
    for i in range(len(cfg["TEST_TRANSFORM"])):
        transform_list.append(create_object_from_dict(cfg["TEST_TRANSFORM"][i]))
    test_transforms = Compose(transform_list)

test_dataset = Dataset(test_data, test_transforms)

mip_transform = Compose([
    LoadImaged(keys=["seg_mip"]),
    EnsureChannelFirstd(keys=["seg_mip"]),
    Orientationd(keys=["seg_mip"], axcodes="LPS"),
    SqueezeDimd(keys=["seg_mip"], dim=-1),
    DeleteItemsd(keys=["pre", "post_1", "post_2", "post_3", "post_3", "seg", "breast"]),
])

mip_dataset = Dataset(test_data, mip_transform)

device = "cuda:1"

model = create_object_from_dict(cfg["MODEL"]).to(device)
model.load_state_dict(torch.load("/cache/marco/experiments/vessel_segmentation/99990-split/42/best_model.pth", weights_only=True, map_location=device))

metric_mip = DiceMetric(reduction = "none")
metric_processed_mip = DiceMetric(reduction = "none")
metric_union = DiceMetric(reduction = "none")
metric_processed_union = DiceMetric(reduction = "none")
loss_mip = DiceLoss()
loss_processed_mip = DiceLoss()
loss_union = DiceLoss()
loss_processed_union = DiceLoss()
losses_mip = []
losses_processed_mip = []
losses_union = []
losses_processed_union = []

ids = []
for i in range(len(test_dataset)):
# for i in [0, 1]:
    print(i)
    ids.append(test_data[i]["pre"].split("/")[6])
    img = test_dataset[i]["img"].to(device)
    seg = test_dataset[i]["seg"]
    seg_mip = mip_dataset[i]["seg_mip"]
    label = union = torch.logical_or(seg_mip, seg.max(3).values).int()
    with torch.no_grad():
        val_outputs = sliding_window_inference(img[None, ...], [96, 96, 96], 4, model)

    if val_outputs.shape[1] == 2:
        pred = AsDiscrete(argmax=True)(val_outputs[0])[0]
        prob = Sigmoid()(val_outputs[0][1])
    elif val_outputs.shape[1] == 1:
        prob = val_outputs[0][0]
        pred = (prob > 0.5).int()
    else:
        print(f"Number of output channels different from 1 or 2")
        pred = None
        prob = None

    pred_mip = pred.max(2).values
    prob_mip = prob.max(2).values

    # prob_mip  = cv.resize(prob_mip.detach().cpu().numpy(), (200, 200))
    prob_mip = prob_mip.detach().cpu().numpy()

    # prob_mip[prob_mip < 0.35] = 0.0
    # plt.imshow(prob_mip, cmap="gray")
    # plt.show()

    prob_mip_processed, losses = postproc_ph(prob_mip, 200, 2, 3, 300, verbose=False, print_per_epoch=1)


    # max_features_list = [25, 50, 100, 200]
    # iters_list = [25, 50, 100, 200]
    # imgs = torch.zeros(len(max_features_list), len(iters_list), *prob_mip.shape)
    #
    # # for i, (max_features, iters_list) in enumerate(product(max_features_list, iters_list)):
    # for i, max_features in enumerate(max_features_list):
    #     for j, iters in enumerate(iters_list):
    #         print(f'max_features={max_features}, iters={iters}')
    #         # prob_mip_processed, losses = postproc_ph(prob_mip, max_features, 6, 5, iters_list, verbose=False, print_per_epoch=1)
    #         imgs[i, j], losses = postproc_ph(prob_mip, max_features, 6, 5, iters, verbose=False,
    #                                              print_per_epoch=1)
    #         plt.figure(figsize=(4, 4), dpi=500)
    #         plt.subplot(len(max_features_list), len(iters_list), 1+j+i*len(iters_list))
    #         plt.plot(losses)
    # plt.show()
    # plt.figure(figsize=(4, 4), dpi=500)
    # for i, max_features in enumerate(max_features_list):
    #     for j, iters in enumerate(iters_list):
    #         plt.subplot(len(max_features_list), len(iters_list), 1+j+i*len(iters_list))
    #         plt.imshow(imgs[i, j], cmap="gray")
    # plt.show()
    # plt.plot(losses)
    # plt.show()

    metric_mip((torch.tensor(prob_mip)[None, None, ...]>0.5).int(), seg_mip[None, ...])
    metric_processed_mip((prob_mip_processed[None, None, ...]>0.5).int(), seg_mip[None, ...])
    metric_union((torch.tensor(prob_mip)[None, None, ...]>0.5).int(), union[None, ...])
    metric_processed_union((prob_mip_processed[None, None, ...]>0.5).int(), union[None, ...])
    losses_mip.append(loss_mip(torch.tensor(prob_mip)[None, None, ...], seg_mip[None, ...]).item())
    losses_processed_mip.append(loss_processed_mip(prob_mip_processed[None, None, ...], seg_mip[None, ...]).item())
    losses_union.append(loss_union(torch.tensor(prob_mip)[None, None, ...], union[None, ...]).item())
    losses_processed_union.append(loss_processed_union(prob_mip_processed[None, None, ...], union[None, ...]).item())

    plt.figure(figsize=(30, 10))  # 3000x1000 pixels (assuming 100 DPI)

    # First image
    plt.subplot(1, 3, 1)
    plt.imshow(prob_mip, cmap="gray")
    plt.axis("off")  # Remove axes

    # Second image
    plt.subplot(1, 3, 2)
    plt.imshow(prob_mip_processed, cmap="gray")
    plt.axis("off")  # Remove axes

    # Loss plot
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.grid(False)  # Remove grid
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks

    plt.tight_layout(pad=0)  # Minimize spacing
    plt.savefig(os.path.join("/cache/marco/experiments/topo_postprocessing/union_loss_2/", ids[i]))


    with open("/cache/marco/experiments/topo_postprocessing/union_loss_2/res.csv", "w") as f:
        f.write(
            f'id, metric_mip, metric_processed_mip, metric_union, metric_processed_union, loss_mip, loss_processes_mip, loss_union, loss_processed_union\n')
        for i in range(len(ids)):
            f.write(
                f'{ids[i]}, {metric_mip.aggregate()[i].item()}, {metric_processed_mip.aggregate()[i].item()}, {metric_union.aggregate()[i].item()}, {metric_processed_union.aggregate()[i].item()}, {losses_mip[i]}, {losses_processed_mip[i]}, {losses_union[i]}, {losses_processed_union[i]}\n')

# print(metric_mip.aggregate())
# print(metric_processed_mip.aggregate())
# print(metric_union.aggregate())
# print(metric_processed_union.aggregate())
# print(losses_mip)
# print(losses_processed_mip)
# print(losses_union)
# print(losses_processed_union)

# print(metric_mip.aggregate().mean())
# print(metric_processed_mip.aggregate().mean())



# running_tensor = torch.tensor(prob_mip).requires_grad_(True)
#
# # running_tensor = prob_mip.detach().clone().requires_grad_(True)
#
# frames = []
# losses = []
#
# for i in range(500):
#     print(i)
#
#     ph = PH(max_features=100)
#     intervals = ph(running_tensor)
#     connected_components = intervals[:1]
#     loss = TopoLoss()(connected_components, [2])
#
#     loss.backward()
#     losses.append(loss.item())
#
#     print(i, loss.item())
#
#     # if i%50==0:
#     #     plt.imshow(running_tensor.detach().cpu(), cmap="gray")
#     #     plt.show()
#     #     # plt.imshow(running_tensor.grad.detach().cpu(), cmap="gray")
#     #     # plt.show()
#     #     intervals_gd = []
#     #     for i in range(len(intervals)):
#     #         for j in range(len(intervals[i])):
#     #             if (intervals[i][j][0] != 0.0) and ((intervals[i][j][1] != 0.0)):
#     #                 intervals_gd.append((i, (intervals[i][j][0].item(), intervals[i][j][1].item())))
#     #     gd.plot_persistence_barcode(intervals_gd)
#     #     plt.show()
#
#     if i%5==0:
#         frames.append((running_tensor.detach().cpu().numpy()*255).astype("uint8").transpose(1, 0))
#
#
#         # plt.imshow((running_tensor - 10*running_tensor.grad).detach().cpu(), cmap="gray")
#         # plt.show()
#
#     running_tensor = (running_tensor - 1*running_tensor.grad).detach().clone().requires_grad_(True)
#     if running_tensor.max() > 1.0:
#         print("MAX greater than 1")
#     if running_tensor.min() < 0.0:
#         print("MIN lesser than 0")
#
# plt.plot(losses)
# plt.show()
# print(len(frames))
# # imageio.mimsave(os.path.join("/home/marco/test.gif"), frames)