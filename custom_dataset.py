# System
import os

# Pip
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# Custom
import config
from helper import get_preprocess_function, CLASS_ID_LIST


class CustomImagenetDataset(Dataset):
    def __init__(self, train, model_name):
        self.train = train
        self.model_name = model_name

        if train:
            # want   : 0 ~ 1199, 1200 ~ 2399, 2400 ~ 3599
            # actual : 0 ~ 1199, 1300 ~ 2499, 2600 ~ 3799
            self.length = config.TRAIN_DATA_LEN
            self.data_path = config.DATA_PATH
        else:
            # want   : 0 ~ 99, 100 ~ 199, 200 ~ 299
            # actual : 1200 ~ 1299, 2500 ~ 2599, 3800 ~ 3899
            self.length = config.TEST_DATA_LEN
            self.data_path = config.TEST_DATA_PATH

        self.transform = get_preprocess_function(model_name, is_crop=True)

    def __len__(self):
        return self.length * 3

    def __getitem__(self, idx):
        label = idx // self.length

        img_index = f'{idx % self.length}.JPEG'
        class_id = CLASS_ID_LIST[label]
        img_path = os.path.join(self.data_path, class_id, img_index)
        image = Image.open(img_path)

        if self.model_name not in ['RNN', 'LSTM', 'GRU']:
            image = image.convert('RGB')

        image = self.transform(image)

        return (image, label)


class CustomImagenetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, model_name):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = CustomImagenetDataset(train=True, model_name=self.model_name)
        self.train_dataset, self.val_dataset = random_split(dataset, [3300, 300])
        self.test_dataset = CustomImagenetDataset(train=False, model_name=self.model_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=config.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=config.NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=config.NUM_WORKERS)


if __name__ == '__main__':
    dataset = CustomImagenetDataset(train=True)
    train_dataset, val_dataset = random_split(dataset, [3300, 300])

    print('------train----------')
    for i in range(3300):
        train_dataset[i]
    print('------valid----------')
    for i in range(300):
        val_dataset[i]

    print('------test----------')
    test_dataset = CustomImagenetDataset(train=False)
    for i in range(config.TEST_DATA_LEN * 3):
        test_dataset[i]
