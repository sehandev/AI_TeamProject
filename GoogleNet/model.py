import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM



class GoogLeNet(pl.LightningModule):
    #GoogleNet(Inception v1) Class
    def __init__(self, learning_rate) :
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.aux1 = Aux(512)
        self.aux2 = Aux(528)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 3)

        self.lr = learning_rate
        self.loss = F.cross_entropy

    def forward(self, x):
        #The model has nine inception structure, one stem region, and three classifiers.
        #Stem-region(not use inception-module area)
        # In ( N x 3 x 224 x 224 ) ->
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        # -> Out ( N x 192 x 28 x 28 )

        #The boundary of the inception area is divided by maxpool.
        #inception area 1 (1, 2)
        # In ( N x 192 x 28 x 28 ) ->
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        # -> Out ( N x 480 x 14 x 14 )

        # inception area 2 (3 ~ 7)
        # aux1, aux2 = auxiliary classifier 
        # In ( N x 480 x 14 x 14 ) ->
        x = self.inception4a(x)
        aux1 = None
        if self.training:  #aux1, aux2 is used only in trains (not in tests).
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = None
        if self.training:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        # -> Out ( N x 832 x 7 x 7 )

        # inception area 3 (8, 9)
        # In ( N x 832 x 7 x 7 ) ->
        x = self.inception5a(x)
        x = self.inception5b(x)
        # -> Out ( N x 1024 x 7 x 7 )

        #final Classifier
        # In ( N x 1024 x 7 x 7 ) ->
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc3(x)
        # -> Out ( N x 3 )

        return x
    
    # obtains and passes the cross entropy loss between the output of the model and the correct label.
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    # obtains and stores model accuracy and cross entropy loss.
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y_hat = F.softmax(y_hat, dim=1)
        acc = FM.accuracy(y_hat, y)  # In logits, the maximum label matches the actual label.

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    # records accuracy and cross entropy loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0001)

class Inception(nn.Module):
    # Perform 1*1, 3*3, and 5*5 Convolution operations to efficiently extract features
    def __init__( self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        #1*1 Convolution operations
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # Branch 2,3 use an additional 1*1 convolution layer. It has the effect of decreasing computation.
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),      #1*1 Convolution operations
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  #3*3 Convolution operations
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),      # 1*1 Convolution operations
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)  # 5*5 Convolution operations
        )

        # Branch 4 use an additional 1*1 convolution laye after maxpool. 
        # It is for reduce the number of channels.
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Change Dimensions
        return torch.cat((branch1, branch2, branch3, branch4), 1)


class Aux(nn.Module):
    def __init__(self, in_channels):
        super(Aux, self).__init__()
        # Googlenet uses an auxiliary classifier structure in two places 
        # to solve the Vanishing gradient problem.
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 3)

    def forward(self, x):
        #The classifier from the third inception module is called Aux1, and the sixth is called Aux2.
        
        # Aux1 : In ( N x 512 x 14 x 14 ) ->
        # Aux2 : In ( N x 528 x 14 x 14 ) ->
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc3(x)
        # -> Out ( N x 3 )
        return x

class BasicConv2d(nn.Module):
    #Convolution layer class
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)