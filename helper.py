# Standard
import os

# PIP
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom
import config

# Model
from ResNet.model import ResNetModel
from LSTM.model import LSTMModel
from GRU.model import GRUModel
from VGGNet.model import VGGNet
from GoogleNet.model import googlenet


CLASS_IDS = {
    'leopard': 'n02128385',
    'jaguar': 'n02128925',
    'cheetah': 'n02130308',
}

CLASS_NAME_LIST = list(CLASS_IDS.keys())
CLASS_ID_LIST = list(CLASS_IDS.values())


def get_class_id(class_name):
    return CLASS_IDS[class_name]


def early_stopping():
    return EarlyStopping(
        monitor='val_loss',  # 기준으로 삼을 metric
        patience=3,  # epoch 몇 번동안 성능이 향상되지 않으면 stop할지
        verbose=True,  # 출력 yer or no
        mode='min'  # monitor 값이 max or min 중 어디로 향해야 하는지
    )


def get_preprocess_function(model_name, is_crop=True):
    if is_crop:
        crop_size = 224
    else:
        crop_size = 256

    if model_name in ['ResNet50', 'VGGNet', 'GoogLeNet']:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.485), (0.229)),
        ])
    else:
        print('ERROR : No implemented model')
        return

    return preprocess


def get_model(model_name, learning_rate):
    if model_name == 'ResNet50':
        model = ResNetModel(learning_rate)
    elif model_name == 'LSTM':
        model = LSTMModel(224, 1000, 3, 3, learning_rate)
    elif model_name == 'GRU':
        model = GRUModel(224, 1000, 3, 3, learning_rate)
    elif model_name == 'VGGNet':
        model = VGGNet(16, 3, 64, 3, learning_rate)
    elif model_name == 'GoogLeNet':
        model = googlenet(learning_rate)
    else:
        print('ERROR : No implemented model')
        return

    return model
