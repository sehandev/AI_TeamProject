import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

# Custom
from helper import conv1x1, load_model, NUM_LAYERS
from bottleneck import Bottleneck



class ResNet(nn.Module):

    def __init__(
        self,
        num_layer_list,
        num_class,
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # layers
        self.layer1 = self._make_layer(64, num_layer_list[0], 1)
        self.layer2 = self._make_layer(128, num_layer_list[1], 2)
        self.layer3 = self._make_layer(256, num_layer_list[2], 2)
        self.layer4 = self._make_layer(512, num_layer_list[3], 2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # max pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # avarage pooling
        self.fc = nn.Linear(512 * 4, num_class)  # fully connected

        for module in self.modules():
            # print(module)
            if isinstance(module, nn.Conv2d):
                # Init with reference [13] normal distribution
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                # Init weight, bias
                module.weight.data.uniform_(0.0, 1.0)
                module.bias.data.fill_(0)

    def _make_layer(self, channels, num_block, stride):

        # Set downsample
        downsample = nn.Sequential(
            conv1x1(self.in_channels, channels * 4, stride),
            nn.BatchNorm2d(channels * 4),
        )

        # Init layer_list
        layer_list = [
            Bottleneck(self.in_channels, channels, stride, downsample),
        ]

        # Update in_channels
        self.in_channels = channels * 4
        
        # Append blocks
        for _ in range(1, num_block):
            layer_list.append(Bottleneck(self.in_channels, channels, 1, None))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        # x : [1, 3, 224, 224]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out : [1, 64, 112, 112]

        out = self.maxpool(out)
        # out : [1, 64, 56, 56]

        out = self.layer1(out)
        # out : [1, 256, 56, 56]
        out = self.layer2(out)
        # out : [1, 512, 28, 28]
        out = self.layer3(out)
        # out : [1, 1024, 14, 14]
        out = self.layer4(out)
        # out : [1, 2048, 7, 7]

        out = self.avgpool(out)
        # out : [1, 2048, 1, 1]

        out = torch.flatten(out)
        # out : [2048]

        out = self.fc(out)
        # out : [num_class]

        return out

def _resnet(model_name, num_class, is_pretrained):
    model = ResNet(NUM_LAYERS[model_name], num_class)
    if is_pretrained:
        load_model(model_name, model)
    return model
