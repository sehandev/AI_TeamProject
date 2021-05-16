import torch.nn as nn


class LSTM(nn.Module):

    # (batch_size, n, ) torch already know, you don't need to let torch know
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 입력 벡터의 크기
        self.input_size = input_size
        # 출력 벡터의 크기
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            # RNN 층
            num_layers=4,
            # 신경망에 입력되는 텐서의 첫 번째 차원값이 batch_size가 되도록 지정
            batch_first=True,
            # 양방향
            bidirectional=False
        )

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size * 2, hidden_size),
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        y = self.layers(y)
        return y
