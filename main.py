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


def fit_model(train_config, model_name, model, is_tune=False):
    pl.seed_everything(train_config['seed'], workers=True)

    data_module = CustomImagenetDataModule(
        batch_size=config.BATCH_SIZE,
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
        'accelerator' : 'dp',
    }

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)


def tune_model(tune_config, checkpoint_dir=None, model_name=None):
    model = helper.get_model(model_name, learning_rate=tune_config['lr'])

    fit_model(tune_config, model_name, model, is_tune=True)


# hyper-parameter tuning 함수
def run_tune(model_name):
    tune_config = {
        'seed': tune.randint(0, 1000),  # 0부터 1000 사이의 랜덤한 정수값
        'lr': tune.uniform(1e-4, 1e-5), # 0.0001부터 0.00001 사이의 랜덤한 실수값
        'num_epochs': 50,
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
    )

    print(analysis.best_config)


# model을 config.py의 설정으로 1번 학습시키는 함수
def train_model(model_name):
    pl.seed_everything(config.SEED, workers=True)

    model = helper.get_model(model_name, learning_rate=config.LEARNING_RATE)

    data_module = CustomImagenetDataModule(
        batch_size=config.BATCH_SIZE,
        model_name=model_name,
    )

    # training
    trainer_args = {
        'callbacks': [
            helper.early_stopping(),
        ],
        'gpus': config.NUM_GPUS,
        'max_epochs': config.EPOCHS,
        'accelerator': 'dp',
        'weights_summary': 'full',
    }

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)

    # Save model
    torch.save(model.state_dict(), f'{config.PROJECT_PATH}/model/{model_name}.pth')


if __name__ == '__main__':
    # LSTM, GRU, ResNet50, VGGNet, GoogLeNet
    model_name = 'GRU'
    
    run_tune(model_name)  # hyper-parameter tuning
    # train_model(model_name)  # model을 config.py의 설정으로 학습하기
