# Standard
import os

# PIP
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

# Custom
from .helper import conv1x1
from .bottleneck_pl import Bottleneck


class ResNet(pl.LightningModule):

    def __init__(
        self,
        learning_rate,
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.lr = learning_rate

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # layers
        self.layer1 = self.build_layer(64, 3, 1)
        self.layer2 = self.build_layer(128, 4, 2)
        self.layer3 = self.build_layer(256, 6, 2)
        self.layer4 = self.build_layer(512, 3, 2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # max pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # avarage pooling
        self.fc_out = nn.Linear(2048, 3)  # fully connected
        self.loss = F.cross_entropy

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Init weight with reference [3] normal distribution
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # bias는 사용하지 않음
            elif isinstance(module, nn.BatchNorm2d):
                # Init weight, bias
                module.weight.data.fill_(1)
                module.bias.data.fill_(0)
            elif isinstance(module, nn.Linear):
                # Init weight, bias
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0)

    def build_layer(self, channels, num_block, stride):

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

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0001)
