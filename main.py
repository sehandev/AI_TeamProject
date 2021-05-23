import os

import torch
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Custom
from custom_dataset import CustomImagenetDataModule
from helper import get_class_id, save_model, early_stopping, get_preprocess_function, CLASS_NAME_LIST, NUM_LAYERS, get_preprocess_function_RNN
import config

from resnet.resnet import _resnet, ResNet
from LSTM.LSTM import LSTMModel

# ray.init(include_dashboard=False)


def open_image(class_name, index):
    class_id = get_class_id(class_name)
    file_name = f'{index}.JPEG'
    file_path = os.path.join('data', class_id, file_name)

    input_image = Image.open(file_path)

    return input_image


def debug_model(model):
    for m in model.modules():
        print(m)


def train_model(tune_config, checkpoint_dir=None, model_name='resnet50', num_epochs=20):
    pl.seed_everything(tune_config['seed'])

    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        model = _resnet(model_name, config.NUM_CLASS, True, tune_config['lr'])
    elif model_name == 'RNN':
        model = LSTMModel(224, 1000, 10, 3, tune_config['lr'])
    elif model_name == 'VGG':
        print('Not yet VGG')
        return
    elif model_name == 'GoogLeNet':
        print('Not yet GoogLeNet')
        return
    else:
        print('ERROR : No implemented model')
        return

    if model_name == 'RNN':
        dm = CustomImagenetDataModule(batch_size=tune_config['batch_size'], isRNN=True)
    else:
        dm = CustomImagenetDataModule(batch_size=tune_config['batch_size'], isRNN=False)

    metrics = {'loss': 'val_loss', 'acc': 'val_acc'}

    # training
    trainer_args = {
        'callbacks': [
            TuneReportCallback(metrics, on='validation_end'),
            early_stopping(),
        ],
        'gpus': config.NUM_GPUS,
        'max_epochs': num_epochs,
        'progress_bar_refresh_rate': 0,
    }

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, dm)


def run_tune(model_name):
    tune_config = {
        'seed': tune.randint(0, 1000),  # a, b 사이의 정수
        'lr': tune.uniform(1e-4, 1e-5),  # a, b 사이의 소수
        'batch_size': 10,
    }

    trainable = tune.with_parameters(
        train_model,
        model_name=model_name,
        num_epochs=config.EPOCHS,
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

    print(analysis.best_config)


def main(model_name, is_pretrained, class_name, index):
    pl.seed_everything(config.SEED)

    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        model = _resnet(model_name, config.NUM_CLASS, True, config.LEARNING_RATE)
    elif model_name == 'RNN':
        model = LSTMModel(224, 1000, 10, 3, config.LEARNING_RATE)
    elif model_name == 'VGG':
        print('Not yet VGG')
        return
    elif model_name == 'GoogLeNet':
        print('Not yet GoogLeNet')
        return
    else:
        print('ERROR : No implemented model')
        return

    if model_name == 'RNN':
        dm = CustomImagenetDataModule(batch_size=config.BATCH_SIZE, isRNN=True)
    else:
        dm = CustomImagenetDataModule(batch_size=config.BATCH_SIZE, isRNN=False)

    metrics = {'loss': 'val_loss', 'acc': 'val_acc'}

    # training
    trainer_args = {
        'gpus': config.NUM_GPUS,
        'max_epochs': config.EPOCHS,
        # 'progress_bar_refresh_rate' : 0,
    }

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, dm)

    # Test the model with the best weights
    trainer.test()

    # Save model
    trainer.save_checkpoint(f'./model/best_{model_name}.ckpt')

    test_model(model_name, class_name, index)


def test_model(model_name, class_name, index):
    if model_name in ['resnet50', 'resnet101', 'resnet152']:
        model = ResNet.load_from_checkpoint(
            checkpoint_path=f'./model/best_{model_name}.ckpt',
            num_layer_list=NUM_LAYERS[model_name],
            num_class=config.NUM_CLASS,
            learning_rate=0,
        )
    elif model_name == 'RNN':
        model = LSTMModel.load_from_checkpoint(
            checkpoint_path=f'./model/best_{model_name}.ckpt',
            input_dim=224,
            hidden_dim=1000,
            layer_dim=10,
            output_dim=3,
            learning_rate=0,
        )
    elif model_name == 'VGG':
        print('Not yet VGG')
        return
    elif model_name == 'GoogLeNet':
        print('Not yet GoogLeNet')
        return
    else:
        print('ERROR : No implemented model')
        return

    # Test image
    input_image = open_image(class_name, index)

    if model_name == 'RNN':
        preprocess = get_preprocess_function_RNN()
    else:
        preprocess = get_preprocess_function()

    input_tensor = preprocess(input_image)  # [3, 224, 224]
    input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]

    # GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)  # [num_class]

    # Calculate probability_array
    probability_array = torch.nn.functional.softmax(output, dim=1)  # [num_class]

    # Select top k from probability array
    K = 3
    print(f'\n [ {model_name} Top {K} ]')
    topk_array, topk_category_index = torch.topk(probability_array, K)
    topk_array = topk_array[0]
    topk_category_index = topk_category_index[0]
    for i in range(len(topk_array)):
        class_name = CLASS_NAME_LIST[topk_category_index[i]]
        probability = topk_array[i].item() * 100
        print(f'{class_name:<10} : {probability:6.3f}%')


if __name__ == '__main__':
    class_name = 'jaguar'
    index = 10

    # print(f' [ Predict {class_name} - {index} ]')
    # main('resnet50', True, class_name, index)
    # main('RNN', True, class_name, index)
    # run_tune('resnet50')
    run_tune('RNN')
    # test_model('RNN', class_name, index)
