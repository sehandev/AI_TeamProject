from torch import nn

# Custom
from helper import conv1x1, conv3x3


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, channels, stride, downsample
    ):
        super(Bottleneck, self).__init__()

        self.stride = stride
        self.downsample = downsample

        self.conv1 = conv1x1(in_channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = conv1x1(channels, channels * 4, 1)
        self.bn3 = nn.BatchNorm2d(channels * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # 1st block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 2nd block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 3rd block
        out = self.conv3(out)
        out = self.bn3(out)

        # Set identity with downsample
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        # skip connection
        out += identity

        # relu
        return self.relu(out)
