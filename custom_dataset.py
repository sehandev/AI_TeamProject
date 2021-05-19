import os
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import numpy as np

from helper import get_preprocess_function, CLASS_ID_LIST
from config import DATA_PATH, ALL_DATA_LEN, TEST_DATA_LEN, BATCH_SIZE, NUM_WORKERS

class CustomImagenetDataset(Dataset):
    def __init__(self, train):
        if train:
            self.idx_start = 0
            self.length = (ALL_DATA_LEN - TEST_DATA_LEN) * 3
        else:
            self.idx_start = ALL_DATA_LEN - TEST_DATA_LEN
            self.length = TEST_DATA_LEN * 3

        self.data_path = DATA_PATH
        self.class_id_list = CLASS_ID_LIST
        self.transform = get_preprocess_function()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx += self.idx_start

        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = idx // 1300
        img_index = f'{idx % 1300}.JPEG'
        class_id = self.class_id_list[label]
        img_path = os.path.join(self.data_path, class_id, img_index)
        image = Image.open(img_path)
        image = self.transform(image)

        if len(image) == 1:
            image = torch.cat((image, image, image), dim=0)
        
        return (image, label)

class CustomImagenetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        dataset = CustomImagenetDataset(train=True)
        self.train_dataset, self.val_dataset = random_split(dataset, [3300, 300])
        self.test_dataset = CustomImagenetDataset(train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)
