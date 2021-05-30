# Standard
import os
import random

# PIP
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

# Custom
import config

# Model
from ResNet.model import ResNet
from LSTM.model import LSTMModel
from GRU.model import GRUModel
from VGGNet.model import VGGNet
from GoogleNet.model import GoogLeNet


CLASS_IDS = {
    'leopard': 'n02128385',
    'jaguar': 'n02128925',
    'cheetah': 'n02130308',
}

CLASS_NAME_LIST = list(CLASS_IDS.keys())
CLASS_ID_LIST = list(CLASS_IDS.values())


def get_class_id(class_name):
    """동물 이름으로 ImageNet의 class id를 불러오기
    
    Parameters
    ----------
    class_name : str
        동물 이름
    """
    return CLASS_IDS[class_name]


def early_stopping():
    # pytorch lightning의 trainer가 overfitting을 막도록 멈추는 함수

    return EarlyStopping(
        monitor='val_loss',  # 기준으로 삼을 metric
        patience=3,  # epoch 몇 번동안 성능이 향상되지 않으면 stop할지
        verbose=True,  # 출력 yer or no
        mode='min'  # monitor 값이 max or min 중 어디로 향해야 하는지
    )


def get_preprocess_function(model_name, is_crop=True):
    """train과 test에 사용할 image를 전처리하는 함수
    
    Parameters
    ----------
    model_name : str
        model 이름
    is_crop : bool, optional
        image의 가운데 224 x 224를 잘라낼지
        default value is True
    """

    if is_crop:
        # image를 가운데를 기준으로 잘라냄
        resize_size = 256
    else:
        # image를 자르지 않고 그대로 사용함
        resize_size = 224

    if model_name in ['ResNet', 'VGGNet', 'GoogLeNet']:
        preprocess = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        # RNN은 RGB image를 인식하지 못하므로 GrayScale로 바꾸어 준다.
        preprocess = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(224),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
    else:
        print('ERROR : No implemented model')
        return

    return preprocess


def get_model(model_name, learning_rate):
    """model object를 생성하는 함수
    
    Parameters
    ----------
    model_name : str
        model 이름
    learning_rate : float
        model을 학습시킬 때 적용할 learning rate
    """

    if model_name == 'ResNet':
        model = ResNet(learning_rate)
    elif model_name == 'LSTM':
        model = LSTMModel(224, 1000, 3, 3, learning_rate)
    elif model_name == 'GRU':
        model = GRUModel(224, 1000, 3, 3, learning_rate)
    elif model_name == 'VGGNet':
        model = VGGNet(16, 3, 64, 3, learning_rate)
    elif model_name == 'GoogLeNet':
        model = GoogLeNet(learning_rate)
    else:
        print('ERROR : No implemented model')
        return

    return model


def force_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
