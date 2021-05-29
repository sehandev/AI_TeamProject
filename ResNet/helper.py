# PIP
from torch import nn


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
