import cv2 as cv
import gudhi as gd
import numpy as np
import torch
from math import ceil

from persistent_homology import PH, topo_loss_fn

import matplotlib.pyplot as plt

def postproc_ph(pred, max_features, y, lr=1, iters=500, e=None, verbose=False, print_per_epoch=10):
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
        # intervals = intervals[:1]
        loss = loss_fn(intervals, y)
        losses.append(loss.item())
        loss.backward()

        if verbose:
            if i % verbose_step == verbose_step - 1:
                print(f'Iteration {i}')
                tmp = running_tensor.detach().cpu()
                plt.imshow(tmp, cmap="gray")
                plt.show()
                cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1 - tmp).flatten(), dimensions=tmp.shape)
                intervals = cubical_complex.persistence()
                gd.plot_persistence_barcode(intervals)
                plt.show()

        running_tensor = torch.clamp(running_tensor - lr * running_tensor.grad, min=0.0, max=1.0)
        running_tensor = running_tensor.detach().clone().requires_grad_(True)

    return running_tensor.detach(), losses

def update_win(val):
    global current
    current = val
    th = val/100
    _, tmp = cv.threshold(image, 1-th, 1, cv.THRESH_BINARY)
    cv.imshow('win', tmp)


image = cv.imread(r"C:\Users\marco\PycharmProjects\MRI\media\test_ph.png", cv.IMREAD_GRAYSCALE)
image = (255-image)/image.max()
image = image.astype(np.float32)
image = cv.resize(image, (30, 30))

plt.imshow(image, cmap="gray")
plt.show()

cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1-image).flatten(), dimensions=image.shape)

intervals = cubical_complex.persistence()
gd.plot_persistence_barcode(intervals)
plt.show()

diagram0 = cubical_complex.persistence_intervals_in_dimension(0)
gd.plot_persistence_diagram(diagram0)
plt.show()

diagram1 = cubical_complex.persistence_intervals_in_dimension(1)
gd.plot_persistence_diagram(diagram1)
plt.show()

res, loss = postproc_ph(image, 20, [1, 1], 2.0, 40, verbose=True, print_per_epoch=10)

cv.namedWindow("win", cv.WINDOW_NORMAL)

current=0
cv.createTrackbar("threshold", "win", 0, 100, update_win)

update_win(current)
cv.waitKey()