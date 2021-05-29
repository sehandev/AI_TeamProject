import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM


# 모델 출처: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#step-3-create-model-class
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

        # Learning Rate
        self.lr = learning_rate
        # Loss function
        self.loss = F.cross_entropy
        if torch.cuda.is_available():
            pass

    # forward propagation function
    def forward(self, x):
        # color값 제거 [batch_size, color, input_size, input_size] -> [batch_size, input_size, input_size]
        x = x.squeeze()
        if len(list(x.size())) < 3:  # batch_size == 1이면 다시 늘려줌
            x = x.unsqueeze(0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # [layer_dim, 3, 1000]

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # [layer_dim, 3, 1000]

        # 224 time steps
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 224, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 3
        return out

    # code 출처: https://www.secmem.org/blog/2021/01/07/pytorch-lightning-tutorial/
    # 모델의 output과 정답 라벨 사이의 cross entropy loss를 구해서 넘겨주는 함수
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        return loss

    # 모델의 정확도와 cross entropy loss를 구해서 저장하는 함수
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y_hat = F.softmax(y_hat, dim=1)
        acc = FM.accuracy(y_hat, y) # logits에서 최댓값인 라벨이 실제 라벨과 일치하는 비율을 구해줌

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    # 정확도와 cross entropy loss를 기록하는 함수
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        # SGD를 사용하였을 때, loss값이 줄어들지 않아서 AdamW로 함수를 변경하였다.
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0001)
