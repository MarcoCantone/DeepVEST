import torch
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference

device = "cuda:2"

# Create the network object
model = AttentionUnet(
    channels=[32, 64, 128, 256],
    in_channels=2,  # pre- and post-contrast
    out_channels=2, # background and vessels
    spatial_dims=3,
    strides=[2, 2, 2]
)
model = model.to(device)

# Load the model
weights_path = ".../best_model.pth"
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device(device)))
model.eval()

# The model expect a tensor with shape [batch, channel, 96, 96, 96] (trained on subvolumes) where in the channel
# dimension the pre and post contrast sequence are concatenated

# The expect transformation include a reorientation of each volume in the LPS space and a MinMax scaling in the [0, 1]
# range.

# to run the inference on a volume you have to apply the model multiple times on [96, 96, 96] subvolumes until the
# full volume is covered. A simple way to do this is using the function "sliding_window_inference" of monai.

input = torch.rand(4, 2, 512, 512, 128).to(device)
output = sliding_window_inference(input, [96, 96, 96], 4, model)

