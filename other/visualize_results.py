import torch
import os
import matplotlib.pyplot as plt
import numpy as np


# dir = "/cache/marco/experiments/vessel_segmentation/9992-channel_wise_normalization/"
#
#
# workspaces = [os.path.join(dir, x) for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]

workspaces = []

workspaces.append("/cache/marco/experiments/paper_experiments/seed34_AttentionUnet")
workspaces.append("/cache/marco/experiments/paper_experiments/seed34_AttentionUnet_preContrast")
# workspaces.append("/cache/marco/experiments/vessel_segmentation/6-attention_unet")

# workspaces.append("/cache/marco/experiments/vessel_segmentation/98-unet_res_units")
# workspaces.append("/cache/marco/experiments/vessel_segmentation/998-unet_out_channel_1")
# workspaces.append("/cache/marco/experiments/vessel_segmentation/6-attention_unet")

fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
for workspace in workspaces:
    if not os.path.isfile(os.path.join(workspace, 'res')):
        continue
    res = torch.load(os.path.join(workspace, 'res'), map_location=torch.device('cpu'))
    ax.plot(res["metrics"], label=f"{workspace.split('/')[-1]} (max={np.max(res['metrics']):.3f})")
    ax.grid(True)

ax.legend()
fig.show()
plt.close(fig)
