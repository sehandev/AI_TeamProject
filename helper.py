# System
import os

# Pip
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Custom
import config

MODEL_PATHES = {
    'resnet50': '/workspace/model/resnet50.pth',
    'resnet101': '/workspace/model/resnet101.pth',
    'resnet152': '/workspace/model/resnet152.pth',
}

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
        patience=5,  # epoch 몇 번동안 성능이 향상되지 않으면 stop할지
        verbose=True,  # 출력 yer or no
        mode='min'  # monitor 값이 max or min 중 어디로 향해야 하는지
    )

def get_preprocess_function(model_name, is_crop=True):

    if is_crop:
        crop_size = 224
    else:
        crop_size = 256
    
    if model_name in ['resnet50', 'resnet101', 'resnet152', 'VGG', 'GoogLeNet']:
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.225,)),
        ])
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
    elif model_name == 'VGG':
        print('Not yet VGG')
        return
    elif model_name == 'GoogLeNet':
        print('Not yet GoogLeNet')
        return
    else:
        print('ERROR : No implemented model')
        return

    return preprocess

def get_best_checkpoint_path(model_name):
    checkpoint_dir = ''
    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        checkpoint_dir = './resnet/model'
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        checkpoint_dir = './LSTM/model'
    elif model_name == 'VGG':
        print('Not yet VGG')
        return
    elif model_name == 'GoogLeNet':
        print('Not yet GoogLeNet')
        return
    else:
        print('ERROR : No implemented model')
        return

    return f'{checkpoint_dir}/best_{model_name}.ckpt'

def get_checkpoint_callback(model_name):
  # checkpoint : project_path/model/[model name]-epoch=02-val_loss=0.32.ckpt

  checkpoint_dir = os.path.join(config.PROJECT_PATH, 'checkpoint', model_name)
  checkpoint_file_name = model_name + '-{epoch:02d}-{val_loss:.2f}'

  checkpoint_callback = ModelCheckpoint(
      dirpath=checkpoint_dir,
      filename=checkpoint_file_name,
      monitor='val_loss',
      mode='min',
      save_top_k=1,
      save_weights_only=True,
  )

  return checkpoint_callback
