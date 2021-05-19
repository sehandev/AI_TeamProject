import os

import torch
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Custom
from resnet_pl import _resnet
from custom_dataset import CustomImagenetDataModule
from helper import get_class_id, save_model, early_stopping, get_preprocess_function, CLASS_NAME_LIST
import config
from LSTM import LSTM


def open_image(class_name, index):
    class_id = get_class_id(class_name)
    file_name = f'{index}.JPEG'
    file_path = os.path.join("data", class_id, file_name)

    input_image = Image.open(file_path)

    return input_image


def debug_model(model):
    for m in model.modules():
        print(m)


def train_model(tune_config, checkpoint_dir=None, model_name='resnet50', num_epochs=20):
    pl.seed_everything(tune_config['seed'])

    # model = _resnet(model_name, config.NUM_CLASS, True, tune_config['lr'])
    model = LSTM(224, 1000, 3, 3)

    dm = CustomImagenetDataModule(batch_size=tune_config["batch_size"])

    metrics = {"loss": "val_loss", "acc": "val_acc"}

    # training
    trainer_args = {
        'callbacks': [
            TuneReportCallback(metrics, on="validation_end"),
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
        "seed": tune.uniform(0, 1000),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([10])
    }

    trainable = tune.with_parameters(
        train_model,
        model_name=model_name,
        num_epochs=config.EPOCHS,
    )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": config.NUM_CPUS,
            "gpu": config.NUM_GPUS,
        },
        metric="loss",
        mode="min",
        config=tune_config,
        num_samples=config.NUM_SAMPLES,
        name="tune_resnet",
    )

    print(analysis.best_config)


def main(model_name, is_pretrained, class_name, index):
    # seed 고정
    pl.seed_everything(config.SEED)

    # Init model
    resnet_model = _resnet(model_name, config.NUM_CLASS, is_pretrained)

    # Set preprocess function
    preprocess = get_preprocess_function()

    # training
    trainer_args = {
        'callbacks': [early_stopping()],
        'gpus': config.NUM_GPUS,
        'max_epochs': config.EPOCHS,
        # 'progress_bar_refresh_rate' : 20,
        # 'resume_from_checkpoint' : os.path.join('checkpoints', args.checkpoint)
    }

    # Init trainer
    trainer = pl.Trainer(**trainer_args)
    custom_data_module = CustomImagenetDataModule()

    # Train the model
    trainer.fit(resnet_model, custom_data_module)

    # Test the model with the best weights
    trainer.test()

    input_image = open_image(class_name, index)

    preprocess = get_preprocess_function()
    input_tensor = preprocess(input_image)  # [3, 224, 224]
    input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]

    # GPU
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        resnet_model.to('cuda')

    with torch.no_grad():
        output = resnet_model(input_batch)  # [num_class]

    # Calculate probability_array
    probability_array = torch.nn.functional.softmax(output, dim=1)  # [num_class]

    # Select top k from probability array
    K = 3
    print(f'\n [ {model_name} Top {K} ]')
    topk_array, topk_category_index = torch.topk(probability_array, K)
    topk_array = topk_array[0]
    topk_category_index = topk_category_index[0]
    print(topk_array)
    print(topk_category_index)
    for i in range(len(topk_array)):
        class_name = CLASS_NAME_LIST[topk_category_index[i]]
        probability = topk_array[i].item() * 100
        print(f'{class_name:<10} : {probability:6.3f}%')


if __name__ == "__main__":
    class_name = 'cheetah'
    index = 2

    #   print(f' [ Predict {class_name} - {index} ]')
    # main('resnet50', True, class_name, index)
    # main('resnet101', True, class_name, index)
    # main('resnet152', True, class_name, index)
    run_tune('resnet50')
