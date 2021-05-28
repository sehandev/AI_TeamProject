import os

import torch
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from PIL import Image

import config
import helper


def open_image(class_name, index, preprocess):
    class_id = helper.get_class_id(class_name)
    file_name = f'{index}.JPEG'
    file_path = os.path.join('test_data', class_id, file_name)

    image = Image.open(file_path)

    image = image.convert('RGB')
    image = preprocess(image)

    return image


def test(model_name):
    model = helper.get_model(model_name, {'lr' : 0})
    state_dict = torch.load(f'{config.PROJECT_PATH}/model/{model_name}.pth')
    model.load_state_dict(state_dict, strict=False)
    model.eval()


    preprocess = helper.get_preprocess_function(model_name, is_crop=True)

    correct_count = 0
    for class_name in ['cheetah', 'jaguar', 'leopard']:
        class_count = 0
        for index in range(100):

            # Test image
            input_tensor = open_image(class_name, index, preprocess)

            input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]

            # GPU
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)  # [1, num_class]

            output = F.softmax(output, dim=1)
            output = output.squeeze(0)  # [num_class]
            top1_index = torch.argmax(output)

            if helper.CLASS_NAME_LIST[top1_index] == class_name:
                correct_count += 1
                class_count += 1

        print(f'Finish {class_name} - {class_count}%')
    print(f'\nAcc : {correct_count / 3 : .3f}%')

if __name__ == '__main__':
    # LSTM, GRU, ResNet50, VGGNet, GoogLeNet
    model_name = 'GRU'

    test(model_name)
