import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class VGGModel(pl.LightningModule):
    def __init__(self, num_layers, input_size, output_size, num_classes, learning_rate):
        super(VGGModel, self).__init__()
        self.num_classes = num_classes
        self.num_layer = num_layers
        if num_layers == 16:
            self.layer_list = [2, 2, 3, 3, 3]  # +fc 3개
        elif num_layers == 19:
            self.layer_list = [2, 2, 4, 4, 4]  # +fc 3개

        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        # conv층 만들기
        conv_list = []

        for i in range(5):
            conv_list.append(self.make_block(self.input_size, self.output_size, self.layer_list[i]))
            self.input_size = self.output_size
            if i < 3: self.output_size *= 2

        self.conv_layer = nn.Sequential(*conv_list)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.output_size * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def make_block(self, input_size, output_size, num_loop):
        block_list = [nn.Conv2d(input_size, output_size, 3, padding=1), nn.ReLU(inplace=True)]
        for i in range(num_loop - 1):
            block_list.append(nn.Conv2d(output_size, output_size, 3, padding=1))
            block_list.append(nn.ReLU(inplace=True))

        block_list.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.fc_layer(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

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