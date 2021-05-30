import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
# 라이트닝을 적용하면 변수들의 디바이스 위치를 일일이 변경하지 않아도 되고,
# 학습 시에 forward/backward/optimize 과정을 더 간단하게 작업할 수 있다.


# hidden state에 쓰레기값이 저장되지 않도록 가중치를 초기화
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# https://github.com/python-engineer/pytorch-examples/blob/master/pytorch-lightning/lightning.py
# 위 링크에서 pytorch lightning의 전체적인 구조를 참고, 함수 내부를 변경하여 VGG 모델에 적용
class VGGNet(pl.LightningModule):
    # num_layers=16/19, input_size=3(RGB channel#), output_size=64
    # num_classes=3(cheetah, leopard, jaguar), learning_rate = lr
    def __init__(self, num_layers, input_size, output_size, num_classes, learning_rate):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.num_layer = num_layers
        if num_layers == 16:  # VGG16의 conv층 개수
            self.layer_list = [2, 2, 3, 3, 3]  # +fc 3개
        elif num_layers == 19:  # VGG19의 conv층 개수
            self.layer_list = [2, 2, 4, 4, 4]  # +fc 3개

        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()  # cross entropy 손실 함수 사용
        # convolution layer list
        conv_list = []

        # 16/19 모델별 필요한 conv층 생성
        for i in range(5):
            conv_list.append(self.make_block(self.input_size, self.output_size, self.layer_list[i]))
            self.input_size = self.output_size
            if i < 3: self.output_size *= 2

        self.conv_layer = nn.Sequential(*conv_list)
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        self.conv_layer.apply(init_weights)  # init weights

        # fully connected layer
        fc_layer_hidden1 = 2024
        fc_layer_hidden2 = 1024
        self.fc_layer = nn.Sequential(
            nn.Linear(self.output_size * 7 * 7, fc_layer_hidden1),  # 1차원 벡터로 flatten
            nn.ReLU(inplace=True),
            nn.Linear(fc_layer_hidden1, fc_layer_hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(fc_layer_hidden2, num_classes)
        )
        self.fc_layer.apply(init_weights)  # 가중치 초기화

    # conv layer 생성 함수
    def make_block(self, input_size, output_size, num_loop):
        # conv 층 하나마다 relu 함수를 적용한다. inplace=True로 따로 할당하지 않고 그 값에 바로 적용한다.
        # 3x3 filter를 padding=1 stride=1로 적용한다. stride는 default값이 1이다.
        block_list = [nn.Conv2d(input_size, output_size, 3, padding=1), nn.ReLU(inplace=True)]
        for i in range(num_loop - 1):
            block_list.append(nn.Conv2d(output_size, output_size, 3, padding=1))
            block_list.append(nn.ReLU(inplace=True))

        block_list.append(nn.MaxPool2d(2, 2))  # 마지막에는 2x2 stride=2 max pooling을 적용한다.
        return nn.Sequential(*block_list)

    # forward
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 512 * 7 * 7)  # flatten
        x = self.fc_layer(x)
        return x

    # training step으로 학습 자동화
    def training_step(self, batch, batch_nb):
        x, y = batch  # y는 label값
        y_hat = self(x)  # y_hat은 해당 모델로 예상한 결과값
        loss = self.loss(y_hat, y)  # 정해진 label과 prediction값 비교
        return loss

    # https://www.secmem.org/blog/2021/01/07/pytorch-lightning-tutorial/
    # epoch마다 validation loss 확인
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y_hat = F.softmax(y_hatd, dim=1)  # 3개 이상의 분류 모델에는 softmax가 좋다. dim=1로 두번째 차원에 적용
        acc = FM.accuracy(y_hat, y)  # 정확도 측정

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)  # log_dict로 key: value의 딕셔너리 저장

    # SGD 대신 AdamW 사용(loss값 변화 증가)
    def configure_optimizers(self):
        # weight decay 은 가중치 감쇠값. default 는 0이다.
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0001)