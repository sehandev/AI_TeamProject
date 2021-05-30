# Standard
import os

# PIP
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
    """train_config의 설정으로 model을 학습시키는 함수
    
    Parameters
    ----------
    train_config : dict
        model을 학습시킬 변수들이 저장된 dictionary
    model_name : str
        학습시킬 model의 이름
        (data preprocess할 함수를 다르게 적용하기 위해 필요)
    model : GoogLeNet or GRUModel or LSTMModel or ResNet or VGGNet
        학습시킬 model
    is_tune : bool, optional
        hyper-parameter tuning을 위해 실행한 건지 아닌지
        default value is False
    """

    # dataset 불러오기
    data_module = CustomImagenetDataModule(
        batch_size=train_config['batch_size'],
        model_name=model_name,
    )

    # 학습 평가 지표
    metrics = {'loss': 'val_loss', 'acc': 'val_acc'}

    # callback_list : pytorch lighting trainer에 사용할 callback 함수 list
    callback_list = [
        helper.early_stopping()
    ]

    if is_tune:
        # hyper-parameter tuning을 위해 실행했다면
        # Tune report를 활성화해서 자동으로 최적의 parameter를 보고
        callback_list.append(TuneReportCallback(metrics, on='validation_end'))

    # training
    trainer_args = {
        'callbacks': callback_list,
        'gpus': config.NUM_GPUS,  # 몇 개의 GPU를 사용할지
        'max_epochs': train_config['num_epochs'],  # 최대 epoch
        'accelerator' : 'dp',  # 여러 GPU를 사용하는 경우, data pararrel로 학습을 더 빠르게 진행
    }
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)


def tune_model(tune_config, checkpoint_dir=None, model_name=None):
    """tune_config 설정으로 model을 학습시키는 함수
    
    Parameters
    ----------
    tune_config : dict
        model을 학습시킬 변수들이 저장된 dictionary
    checkpoint_dir : str, optional
        학습 중 checkpoint를 저장할 directory path
        default value is None
    model_name : str, optional
        학습시킬 model의 이름
        default value is None
    """

    # random seed 고정
    pl.seed_everything(tune_config['seed'], workers=True)
    helper.force_seed(config.SEED)

    # model 불러오기
    model = helper.get_model(model_name, learning_rate=tune_config['lr'])

    # 설정한 변수들로 model 학습
    fit_model(tune_config, model_name, model, is_tune=True)


def run_tune(model_name):
    """hyper-parameter tuning을 시작하는 함수
    
    Parameters
    ----------
    model_name : str, optional
        학습시킬 model의 이름
        default value is None
    """

    # hyper-parameter tuning을 위한 변수를 지정한 dict
    tune_config = {
        'seed': tune.randint(0, 1000),  # random seed : 0 이상 1000 미만의 랜덤한 정수값
        'lr': tune.loguniform(1e-3, 1e-9),  # learning rate : 1e-3부터 1e-9 사이의 랜덤한 실수값
        'num_epochs': 50,  # epoch : 50
        'batch_size': tune.randint(8, 21),  # 8 이상 21 미만의 랜덤한 정수값
    }

    trainable = tune.with_parameters(
        tune_model,
        model_name=model_name,
    )

    # hyper-parameter tuning
    # loss가 가장 낮은 model을 찾음
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

    # 가장 좋은 model의 설정 출력
    print(analysis.best_config)


def train_model(model_name):
    """model을 config.py의 설정으로 1번 학습시키는 함수
    
    Parameters
    ----------
    model_name : str
        학습시킬 model의 이름
    """

    # random seed 고정
    pl.seed_everything(config.SEED, workers=True)
    helper.force_seed(config.SEED)

    # model 불러오기
    model = helper.get_model(model_name, learning_rate=config.LEARNING_RATE)

    # dataset 불러오기
    data_module = CustomImagenetDataModule(
        batch_size=config.BATCH_SIZE,
        model_name=model_name,
    )

    # model 학습할 변수 설정
    trainer_args = {
        'callbacks': [
            helper.early_stopping(),
        ],
        'gpus': config.NUM_GPUS,
        'max_epochs': config.EPOCHS,  # 
        'accelerator': 'dp',  # 여러 GPU를 사용하는 경우, data pararrel로 학습을 더 빠르게 진행
        'weights_summary': 'full',
    }
    trainer = pl.Trainer(**trainer_args)

    # model 학습
    trainer.fit(model, data_module)

    # Save model
    torch.save(model.state_dict(), f'{config.PROJECT_PATH}/model/{model_name}.pth')


if __name__ == '__main__':
    # LSTM, GRU, ResNet, VGGNet, GoogLeNet
    model_name = 'GRU'
    
    run_tune(model_name)  # hyper-parameter tuning
    # train_model(model_name)  # model을 config.py의 설정으로 학습하기
