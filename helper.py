import torch
from torch import nn
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

MODEL_PATHES = {
    'resnet50': '/workspace/model/resnet50.pth',
    'resnet101': '/workspace/model/resnet101.pth',
    'resnet152': '/workspace/model/resnet152.pth',
}

NUM_LAYERS = {
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

CLASS_IDS = {
    'leopard': 'n02128385',
    'jaguar': 'n02128925',
    'cheetah': 'n02130308',
}

CLASS_NAME_LIST = list(CLASS_IDS.keys())
CLASS_ID_LIST = list(CLASS_IDS.values())


def conv3x3(in_channels, out_channels, stride):
    # 3x3 convolution with padding

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=1,
        bias=False,
        dilation=1
    )


def conv1x1(in_channels, out_channels, stride):
    # 1x1 convolution

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def load_model(model_name, model):
    state_dict = torch.load(MODEL_PATHES[model_name])
    model.load_state_dict(state_dict, strict=False)
    model.eval()


def save_model(model_name, model):
    torch.save(model.state_dict(), MODEL_PATHES[model_name])


def get_class_id(class_name):
    return CLASS_IDS[class_name]


def early_stopping():
    return EarlyStopping(
        monitor='val_loss',  # 기준으로 삼을 metric
        patience=5,  # epoch 몇 번동안 성능이 향상되지 않으면 stop할지
        verbose=True,  # 출력 yer or no
        mode='min'  # monitor 값이 max or min 중 어디로 향해야 하는지
    )


def get_preprocess_function():
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.485,), (0.225,)),
    ])

    return preprocess


def get_preprocess_function_RNN():
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.Normalize((0.485,), (0.225,)),
    ])

    return preprocess