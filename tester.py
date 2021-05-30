# Standard
import os

# PIP
import torch
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from PIL import Image

# Custom
import config
import helper


def open_image(class_name, index, preprocess):
    """image를 불러오는 함수
    
    Parameters
    ----------
    class_name : str
        불러올 동물 이름
    index : int
        불러올 data의 index
    preprocess : function
        image에 적용할 preprocess 함수
    """

    class_id = helper.get_class_id(class_name)
    file_name = f'{index}.JPEG'
    file_path = os.path.join('test_data', class_id, file_name)

    # file_path에서 image 열기
    image = Image.open(file_path)

    # Grayscale image를 RGB로 변환
    image = image.convert('RGB')

    # image 전처리
    image = preprocess(image)

    return image


def test(model_name):
    """model을 테스트하는 함수
    
    Parameters
    ----------
    model_name : str
        테스트할 model 이름
    """

    # model 불러오기
    model = helper.get_model(model_name, {'lr' : 0})
    state_dict = torch.load(f'{config.PROJECT_PATH}/model/{model_name}.pth')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 적용할 전처리 함수 불러오기
    preprocess = helper.get_preprocess_function(model_name, is_crop=True)

    correct_count = 0  # 전체에서 맞춘 개수
    for class_name in ['cheetah', 'jaguar', 'leopard']:
        class_count = 0  # 해당 동물에 대해 맞춘 개수
        for index in range(100):

            # image 불러오기
            input_tensor = open_image(class_name, index, preprocess)

            # batch처럼 적용하기 위해 차원을 한 단계 늘림
            input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]

            # GPU 적용
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            # model로 image가 어떤 동물인지 예측
            with torch.no_grad():
                output = model(input_batch)  # [1, num_class]

            # output을 통해 각 동물을 몇%의 확률로 예측했는지 확인
            output = F.softmax(output, dim=1)
            output = output.squeeze(0)  # [num_class]

            # 가장 높은 확률로 예측한 동물의 index
            top1_index = torch.argmax(output)

            # 정답 확인
            if helper.CLASS_NAME_LIST[top1_index] == class_name:
                correct_count += 1
                class_count += 1

        print(f'Finish {class_name} - {class_count}%')
    print(f'\nAcc : {correct_count / 3 : .3f}%')


if __name__ == '__main__':
    for model_name in ['LSTM', 'GRU', 'ResNet', 'VGGNet', 'GoogLeNet']:
        print(f'== {model_name} ==')
        test(model_name)
