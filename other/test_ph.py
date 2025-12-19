import gudhi as gd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch import nn
from persistent_homology import PH, topo_loss_fn

def update_win(val):
    global current
    current = val
    th = val/100
    _, tmp = cv.threshold(image, 1-th, 1, cv.THRESH_BINARY)
    cv.imshow('win', tmp)

def TopoGrad(S, k, e, atol = 1e-8):
    # create empty matrix for storing pixelwise gradient
    G = np.zeros_like(S)
    dims = len(S.shape)
    for t in range(k):
        # compute persistence homology
        cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1 - S).flatten(), dimensions=S.shape)
        H = cubical_complex.persistence()
        H = [[x[1] for x in H if x[0] == dim] for dim in range(dims)]
        # TODO: generalize the desired topology
        # select birth and death filtration value based on desired topology
        pb, pd = H[1][0] # rank 1, 0 element (the first, longest features)
        if pb > e:
            pixels = np.where(np.isclose(S, 1-pb, atol=atol))
            S[pixels] = 1
            G[pixels] = -1
        if pd < 1 - e:
            pixels = np.where(np.isclose(S, 1-pd, atol=atol))
            S[pixels] = 0
            G[pixels] = 1
    return G


def TopoGradCombined(S, k, e, rank, num_features, incourage=True, discourage=True, atol=1e-8):
    # create empty matrix for storing pixelwise gradient
    G = np.zeros_like(S)
    dims = len(S.shape)
    for t in range(k):
        # compute persistence homology
        cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1 - S).flatten(), dimensions=S.shape)
        H = cubical_complex.persistence()
        H = [[x[1] for x in H if x[0] == dim] for dim in range(dims)]

        # select birth and death filtration value based on desired topology
        incourage_features = H[rank][:num_features] # rank 1, 0 element (the first, longest features)
        discourage_features = H[rank][num_features:]
        if incourage:
            for pb, pd in incourage_features:
                if pb > e:
                    pixels = np.where(np.isclose(S, 1-pb, atol=atol))
                    S[pixels] = 1
                    G[pixels] = -1
                if pd < 1 - e:
                    pixels = np.where(np.isclose(S, 1-pd, atol=atol))
                    S[pixels] = 0
                    G[pixels] = 1
        if discourage:
            for pb, pd in discourage_features:
                if pb > e:
                    pixels = np.where(np.isclose(S, 1-pb, atol=atol))
                    S[pixels] = 0
                    G[pixels] = 1
                if pd < 1 - e:
                    pixels = np.where(np.isclose(S, 1-pd, atol=atol))
                    S[pixels] = 1
                    G[pixels] = -1
    return G

# a = torch.tensor([[0, 0, 0, 0, 0],
# [0, 0.8, 0.8, 0.3, 0],
# [0, 0.8, 0, 0.3, 0],
# [0, 0, 0.3, 0.55, 0],
# [0, 0, 0, 0, 0]], requires_grad=True)
#
#
# max_features = 10
#
# persistence = PH(max_features)
#
# c = persistence(a)
#
# d = c[0, 1, 0]
#
# d = c.sum()
#
# d.backward()
#
# exit()

image = cv.imread("C:\\Users\\marco\\PycharmProjects\\MRI\\media\\test_ph.png", cv.IMREAD_GRAYSCALE)
#image = torch.nn.Sigmoid()(torch.load("C:\\Users\\marco\\Desktop\\pred_example", map_location="cpu")[0, 1]).numpy()
image = (255-image)/image.max()
image = cv.resize(image, (30, 30))

plt.imshow(image, cmap="gray")
plt.show()

# max_features = 5
# persistence = PH(max_features)
# input = torch.tensor(image, requires_grad=True, dtype=torch.float)
# intervals = persistence(input)
# intervals_gd = []
# for i in range(len(intervals)):
#     for j in range(len(intervals[i])):
#         if (intervals[i][j][0] != 0.0) and ((intervals[i][j][1] != 0.0)):
#             intervals_gd.append((i, (intervals[i][j][0].item(), intervals[i][j][1].item())))
# gd.plot_persistence_barcode(intervals_gd)
# plt.show()

cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1-image).flatten(), dimensions=image.shape)
intervals1 = cubical_complex.persistence()  # Compute persistence
gd.plot_persistence_barcode(intervals1)
plt.show()

diagram = cubical_complex.persistence_intervals_in_dimension(0)  # H0 features
gd.plot_persistence_diagram(diagram)
plt.show()

# prob = image.copy()
#
# grad = TopoGrad(prob, 10, 0.001)
# plt.imshow(grad, cmap="gray")
# plt.show()
#
# plt.imshow(image-0.2*grad, cmap="gray")
# plt.show()
#
# for _ in range(5):
#
#     intervals = persistence(input)
#
#
#     loss_fn = TopoLoss()
#     loss = loss_fn(intervals, [1, 1])
#     loss.backward()
#     grad1 = input.grad
#     # plt.imshow(grad1, cmap="gray")
#     # plt.show()
#
#     input = torch.tensor(input-0.2*grad1, requires_grad=True, dtype=torch.float)
#
# intervals_gd = []
# for i in range(len(intervals)):
#     for j in range(len(intervals[i])):
#         if (intervals[i][j][0] != 0.0) and ((intervals[i][j][1] != 0.0)):
#             intervals_gd.append((i, (intervals[i][j][0].item(), intervals[i][j][1].item())))
# gd.plot_persistence_barcode(intervals_gd)
# plt.show()

# plt.imshow(image.detach().numpy(), cmap="gray")
# plt.show()

cv.namedWindow("win", cv.WINDOW_NORMAL)

current=0
cv.createTrackbar("threshold", "win", 0, 100, update_win)

update_win(current)
cv.waitKey()