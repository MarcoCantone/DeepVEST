from torch import nn
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from config import get_class_by_path


def create_model(class_name, model_path, device="cpu", **kwargs):
    cls = get_class_by_path(class_name)
    model = cls(**kwargs).to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return model


class SWNet(nn.Module):
    def __init__(self, backbone):
        super(SWNet, self).__init__()

        self.backbone = backbone

    def forward(self, x):

        x = sliding_window_inference(x, [96, 96, 96], 2, self.backbone)

        return x


class LocalGlobal(nn.Module):
    def __init__(self, backbone, globalnet):
        super(LocalGlobal, self).__init__()
        self.backbone = backbone
        for params in self.backbone.parameters():
            params.requires_grad = False

        self.globalnet = globalnet

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            x += sliding_window_inference(x, [96, 96, 96], 4, self.backbone)

        return self.sigmoid(self.globalnet(x))


class AddSigmoid(nn.Module):
    def __init__(self, net):
        super(AddSigmoid, self).__init__()
        self.net = net
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x))
