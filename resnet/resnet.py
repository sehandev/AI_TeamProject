import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

# Custom
from helper import conv1x1, load_model, NUM_LAYERS
from config import LEARNING_RATE, MOMENTUM, WEIGHT_DECAY
from resnet.bottleneck_pl import Bottleneck


class ResNet(pl.LightningModule):

    def __init__(
            self,
            num_layer_list,
            num_class,
            learning_rate,
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.lr = learning_rate

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
        # self.fc = nn.Linear(512 * 4, 1000)  # fully connected
        # self.fc1 = nn.Linear(1000, num_class)  # fully connected
        self.fc_out = nn.Linear(512 * 4, num_class)  # fully connected
        self.loss = F.cross_entropy

        for module in self.modules():
            # print(module)
            if isinstance(module, nn.Conv2d):
                # Init with reference [batch_size3] normal distribution
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
        # x : [batch_size, 3, 224, 224]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out : [batch_size, 64, 112, 112]

        out = self.maxpool(out)
        # out : [batch_size, 64, 56, 56]

        out = self.layer1(out)
        # out : [batch_size, 256, 56, 56]
        out = self.layer2(out)
        # out : [batch_size, 512, 28, 28]
        out = self.layer3(out)
        # out : [batch_size, 1024, 14, 14]
        out = self.layer4(out)
        # out : [batch_size, 2048, 7, 7]

        out = self.avgpool(out)
        # out : [batch_size, 2048, 1, 1]

        out = torch.flatten(out, 1)
        # out : [batch_size, 2048]

        # out = self.fc(out)
        # out : [batch_size, 1000]

        # out = self.fc1(out)
        # out : [1000, num_class]

        out = self.fc_out(out)
        # out : [batch_size, 3]

        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


def _resnet(model_name, num_class, is_pretrained, learning_rate):
    model = ResNet(NUM_LAYERS[model_name], num_class, learning_rate)
    if is_pretrained:
        load_model(model_name, model)
    return model
