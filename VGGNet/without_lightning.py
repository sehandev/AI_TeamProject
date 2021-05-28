import torch, gc
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import math
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./drive/MyDrive/runs/test")  # for tensorboard


# 이미지 전처리
def get_preprocess_function():
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 256x256로 리사이징
        transforms.CenterCrop(224),  # 가운데 224x224로 크롭
        transforms.ToTensor(),
        # pytorch 공식 홈페이지 (https://pytorch.org/vision/stable/transforms.html)
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return preprocess


# 디바이스 설정. 연산에 사용되는 값들이 같은 디바이스에 위치해야 한다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 비교를 위해 seed값 고정
torch.manual_seed(1234)
torch.cuda.manual_seed_all(4567)

# 하이퍼 파라미터
classes = ('leopard', 'jaguar', 'cheetah')
transform = get_preprocess_function()
num_epochs = 20
learning_rate = 1e-4
batch_size = 120


# train dataset 경로 설정
# imagefoder를 사용하면 해당 디렉토리의 하위 폴더들을 하나의 class로 인식
train_dataset = ImageFolder(root='./TrainData', transform=transform)
# dataloader로 shuffle하여 랜덤하게 batch size씩 분리
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
total_samples = len(train_dataset)  # 3600 = 1200 * 3(class)
# (데이터 전체 크기)%(batch 크기) != 0인 경우 ceil로 올림
n_iterations = math.ceil(total_samples / batch_size)


class VGGNet(nn.Module):
    # num_layers=16/19, input_size=3(RGB channel#), output_size=64(VGG 구조)
    # num_classes=3(cheetah, leopard, jaguar)
    def __init__(self, num_layers, input_size, output_size, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.num_layer = num_layers
        if num_layers == 16:  # VGG16의 conv층 개수
            self.layer_list = [2, 2, 3, 3, 3, 3]
        elif num_layers == 19:  # VGG19의 conv층 개수
            self.layer_list = [2, 2, 4, 4, 4, 3]

        self.input_size = input_size
        self.output_size = output_size

        conv_list = []
        # 모델별 conv층만큼 conv_list 만들어 sequential로 변환
        for i in range(5):
            conv_list.append(self.make_block(self.input_size, self.output_size, self.layer_list[i]))
            self.input_size = self.output_size
            if i < 3: self.output_size *= 2

        self.conv_layer = nn.Sequential(*conv_list)

        # fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Linear(self.output_size * 7 * 7, 4096),  # 1차원 벡터로 flatten
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
            # torch.nn.CrossEntropyLoss() criterion이 LogSoftMax와 NULLoss를 함께 포함하므로 마지막 softmax 생략
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        )

        # conv layer 하나당 relu 함수 적용, 마지막에 max pooling 적용
        def make_block(input_size, output_size, num_loop):
            block_list = [nn.Conv2d(input_size, output_size, 3, padding=1), nn.ReLU(inplace=True)]
            for i in range(num_loop - 1):
                block_list.append(nn.Conv2d(output_size, output_size, 3, padding=1))
                block_list.append(nn.ReLU(inplace=True))

            block_list.append(nn.MaxPool2d(2, 2))  # 2x2 stride=2 max pooling
            return nn.Sequential(*block_list)

    # forward 연산
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 512 * 7 * 7)  # flatten
        x = self.fc_layer(x)
        return x


model = VGGNet(16, 3, 64, 3).to(device)  # VGG16 모델 생성, device 위치로 이동
# model = VGGNet(19, 3, 64, 3).to(device) # VGG 19 모델 생성
criterion = nn.CrossEntropyLoss()  # 손실 함수 cross entropy 사용
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 경사하강법 최적화
n_total_steps = len(train_loader)

# epoch만큼 반복하며 학습
for epoch in range(num_epochs):
    # batch size만큼 반복한다
    for i, (inputs, labels) in enumerate(train_loader):  # index, input data, label
        inputs = inputs.to(device)  # cpu/gpu로 이동
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 손실값 계산

        # backward
        loss.backward()

        # 최적화
        optimizer.step()
        optimizer.zero_grad()  # 값이 누적되므로 초기화

        # 10번째마다 epoch, step, loss값 출력
        if (i + 1) % 10 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1} / {n_total_steps}, loss = {loss.item():.4f}')

        # 불필요하게 할당된 CUDA 용량 확보
        gc.collect()
        torch.cuda.empty_cache()

# print("Training Finish")

# save model
MODEL_PATH = './drive/MyDrive/Data/Model/VGG16.pt'  # VGG16의 경우
torch.save(model.state_dict(), MODEL_PATH)

# test data load
test_dataset = ImageFolder(root='./drive/MyDrive/Data/TestData', transform=transform)
# test data는 class별로 100장이므로 batch_size=100, 클래스별 결과를 출력하기 위해서 shuffle=False
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 저장된 model load
loaded_model = VGGNet(16, 3, 64, 3).to(device)
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.eval()  # 평가하는 과정에서 모든 노드를 사용하겠다는 의미

with torch.no_grad():  # test 과정은 가중치에 영향을 주지 않아야 한다
    total_acc = 0  # 종합 정확도
    for images, labels in test_loader:
        n_correct = 0
        n_samples = 0
        images = images.to(device)
        labels = labels.to(device)

        outputs = loaded_model(images)  # load한 모델로 test data 연산 수행
        _, predictions = torch.max(outputs, 1)  # max함수는 value, index 값 리턴
        n_samples += labels.shape[0]  # 현재 batch에서 sample의 개수를 누적
        n_correct += (predictions == labels).sum().item()  # 예측값과 label 같은 개수의 합

        acc = 100.0 * n_correct / n_samples
        total_acc += acc
        print(f'{classes[labels[0]]} accuracy = {acc}')  # 클래스별 정확도 출력

    print("total accuracy = ", total_acc / 3)  # 종합 정확도 출력
