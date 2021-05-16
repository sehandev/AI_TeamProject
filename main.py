import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import cv2
import LSTM

# 표범 학습 데이터
train_leopard_path = './train/n02128385'
train_leopard = os.listdir(train_leopard_path)
train_leopard.sort(key=lambda fname: int(fname.split('_')[1].split('.')[0]))
# 재규어 학습 데이터
train_jaguar_path = './train/n02128925'
train_jaguar = os.listdir(train_jaguar_path)
train_jaguar.sort(key=lambda fname: int(fname.split('_')[1].split('.')[0]))
# 치타 학습 데이터
train_cheetah_path = './train/n02130308'
train_cheetah = os.listdir(train_cheetah_path)
train_cheetah.sort(key=lambda fname: int(fname.split('_')[1].split('.')[0]))

test_data = []

input_size = 50
hidden_size = 3

model = LSTM.LSTM(input_size, hidden_size)

# loss & optimizer setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

img = []

for i in range(0, 5): # len(train_leopard)):
    img_name = train_leopard_path + '/' + train_leopard[i]
    img.append(cv2.imread(img_name, cv2.IMREAD_COLOR))
img = torch.Tensor(img)

print(img)

for i in range(0, len(img)):
    model.train()
    outputs = model()
