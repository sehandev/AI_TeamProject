import torch
from torch import nn


MODEL_PATHES = {
    'resnet50' : '/workspace/resnet/model/resnet50.pth',
    'resnet101' : '/workspace/resnet/model/resnet101.pth',
    'resnet152' : '/workspace/resnet/model/resnet152.pth',
}

NUM_LAYERS = {
    'resnet50' : [3, 4, 6, 3],
    'resnet101' : [3, 4, 23, 3],
    'resnet152' : [3, 8, 36, 3],
}


def conv3x3(in_channels, out_channels, stride):
    # 3x3 convolution with padding

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False,
        dilation=1
    )

def conv1x1(in_channels, out_channels, stride):
    # 1x1 convolution

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

def load_model(model_name, model):
  state_dict = torch.load(MODEL_PATHES[model_name])
  model.load_state_dict(state_dict, strict=False)
  model.eval()

  return model