import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Custom
from custom_dataset import CustomImagenetDataModule, CustomImagenetDataset
import config
import helper

# Model
from resnet.resnet import _resnet, ResNet, get_resnet_layer
from LSTM.LSTM import LSTMModel
from GRU.GRU import GRUModel
from VGGNet.VGGNet import VGGNet
from GoogleNet.googlenet import googlenet
from GoogleNet.googlenet import GoogLeNet


# image를 불러오는 함수
def open_image(class_name, index, preprocess):
    class_id = helper.get_class_id(class_name)
    file_name = f'{index}.JPEG'
    file_path = os.path.join('test_data', class_id, file_name)

    image = Image.open(file_path)

    image = image.convert('RGB')
    image = preprocess(image)

    #if len(image) == 1:
    #    image = torch.cat((image, image, image), dim=0)

    return image


# model class 생성하는 함수
def get_model(model_name, model_config):
    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        model = _resnet(model_name, model_config['lr'])
    elif model_name in ['RNN', 'LSTM']:
        model = LSTMModel(224, 1000, 3, 3, model_config['lr'])
    elif model_name == 'GRU':
        model = GRUModel(224, 1000, 10, 3, model_config['lr'])
    elif model_name == 'VGGNet':
        model = VGGNet(16, 3, 64, 3, model_config['lr'])
    elif model_name == 'GoogLeNet':
        model = googlenet(model_config['lr'])
    else:
        print('ERROR : No implemented model')
        return

    return model


def fit_model(train_config, model_name, model, is_tune=False):
    pl.seed_everything(train_config['seed'])

    data_module = CustomImagenetDataModule(
        batch_size=train_config['batch_size'],
        model_name=model_name,
    )

    metrics = {'loss': 'val_loss', 'acc': 'val_acc'}

    callback_list = [
        helper.early_stopping()
    ]

    if is_tune:
        callback_list.append(TuneReportCallback(metrics, on='validation_end'))

    # training
    trainer_args = {
        'callbacks': callback_list,
        'gpus': config.NUM_GPUS,
        'max_epochs': train_config['num_epochs'],
        'progress_bar_refresh_rate': 100,
    }

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)


def tune_model(tune_config, checkpoint_dir=None, model_name=None):
    model = get_model(model_name, tune_config)

    fit_model(tune_config, model_name, model, is_tune=True)


# hyper-parameter tuning 함수
def run_tune(model_name):
    tune_config = {
        'seed': tune.randint(0, 1000),  # 0부터 1000 사이의 랜덤한 정수값
        'lr': tune.uniform(1e-4, 1e-5), # 0.0001부터 0.00001 사이의 랜덤한 실수값
        'batch_size': 10,
        'num_epochs': 20,
    }

    trainable = tune.with_parameters(
        tune_model,
        model_name=model_name,
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            'cpu': config.NUM_CPUS,
            'gpu': config.NUM_GPUS,
        },
        metric='loss',
        mode='min',
        config=tune_config,
        num_samples=config.NUM_SAMPLES,
        name=f'tune_{model_name}',
        # resume=True,
    )

    # print(analysis.best_config)


# model을 config.py의 설정으로 1번 학습시키는 함수
def train_model(model_name):
    pl.seed_everything(config.SEED)

    model_config = {
        'lr': config.LEARNING_RATE,
    }
    model = get_model(model_name, model_config)

    data_module = CustomImagenetDataModule(
        batch_size=config.BATCH_SIZE,
        model_name=model_name,
    )

    # training
    trainer_args = {
        'callbacks': [
            helper.early_stopping(),
            helper.get_checkpoint_callback(model_name),
        ],
        'gpus': config.NUM_GPUS,
        'max_epochs': config.EPOCHS,
        'accelerator' : "dp",
        # 'progress_bar_refresh_rate' : 0,
    }

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)

    # Save model
    trainer.save_checkpoint(helper.get_best_checkpoint_path(model_name))


# test data 300개로 정확도 확인하는 함수
def test_model(model_name):
    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        model = ResNet.load_from_checkpoint(
            checkpoint_path=helper.get_best_checkpoint_path(model_name),
            num_layer_list=get_resnet_layer(model_name),
            learning_rate=0,
        )
    elif model_name in ['RNN', 'LSTM']:
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=helper.get_best_checkpoint_path(model_name),
            input_dim=224,
            hidden_dim=1000,
            layer_dim=3,
            output_dim=3,
            learning_rate=0,
        )
    elif model_name == 'GRU':
        model = GRUModel.load_from_checkpoint(
          checkpoint_path=helper.get_best_checkpoint_path(model_name),
          input_dim=224,
          hidden_dim=1000,
          layer_dim=10,
          output_dim=3,
          learning_rate=0,
        )
    elif model_name == 'VGG':
        model = VGGNet.load_from_checkpoint(
          checkpoint_path=helper.get_best_checkpoint_path(model_name),
          input_size=224,
          output_size=10,
          num_classes=3,
          learning_rate=0,
        )
    elif model_name == 'GoogLeNet':
        model = GoogLeNet.load_from_checkpoint(
          checkpoint_path='./GoogleNet/model/best_GoogLeNet.ckpt',
          input_size=224,
          output_size=10,
          num_classes=3,
          learning_rate=0,
        )
    else:
        print('ERROR : No implemented model')
        return

    print(f'SUCCESS : load model {model_name} from checkpoint')

    preprocess = helper.get_preprocess_function(model_name, is_crop=True)

    correct_count = 0
    for class_name in ['cheetah', 'jaguar', 'leopard']:
        class_count = 0
        for index in range(100):

            # Test image
            input_tensor = open_image(class_name, index, preprocess)
            input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224] [batch_size, color, input_size, input_size]

            # GPU
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)  # [1, num_class]
            output = output.squeeze(0)  # [num_class]
            top1_index = torch.argmax(output)

            if helper.CLASS_NAME_LIST[top1_index] == class_name:
                correct_count += 1
                class_count += 1

            # Select top k from probability array
            # K = 3
            # print(f'\n [ {model_name} Top {K} ]')
            # topk_array, topk_category_index = torch.topk(output, K)
            # for i in range(len(topk_array)):
            #   class_name = CLASS_NAME_LIST[topk_category_index[i]]
            #   probability = topk_array[i].item() * 100
            #   print(f'{class_name:<10} : {probability:6.3f}%')
        print(f'Finish {class_name} - {class_count}%')
    print(f'\nAcc : {correct_count / 3 : .3f}%')


if __name__ == '__main__':
    # LSTM, GRU, resnet50, VGGNet, GoogLeNet
    model_name = 'GoogLeNet'
    train_model(model_name)  # model을 config.py의 설정으로 1번 학습하기
    run_tune(model_name)  # hyper-parameter tuning

    test_model(model_name)  # test data 300개로 정확도 확인
