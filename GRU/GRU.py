import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


class LSTMModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, learning_rate):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.lr = learning_rate
        self.loss = F.cross_entropy
        if torch.cuda.is_available():
            pass

    def forward(self, x):
        x = x.squeeze()  # [10, 224, 224] [batch_size, input_size, input_size]
        if x.shape[0] < 3:  # batch_size == 1이면 다시 늘려줌
          x = x.unsqueeze(0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # [layer_dim, 3, 1000]

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # [layer_dim, 3, 1000]

        # One time step
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

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
