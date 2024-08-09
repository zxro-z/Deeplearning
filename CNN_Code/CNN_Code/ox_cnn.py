from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision 
from torchvision import transforms
import torch.nn as nn


DEVICE = torch.device('cpu')  # 기기 설정: CPU 사용
BATCH_SIZE = 10  # 배치 크기 설정
EPOCHS = 30  # 에포크 설정

def prepare_data(root_train, root_test, batch_size, device):
    # 데이터 변환 정의
    trans = transforms.Compose([
        transforms.Resize((100, 100)),  # 이미지 크기 조정
        transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지 좌우 반전
        transforms.RandomRotation(15),  # 랜덤하게 이미지 회전
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화
    ])

    # 학습 및 테스트 데이터셋 불러오기
    trainset = torchvision.datasets.ImageFolder(root=root_train, transform=trans)
    testset = torchvision.datasets.ImageFolder(root=root_test, transform=trans)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)  # 입력 채널을 3으로 수정
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=9680, out_features=50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=3, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = x.view(-1, 9680)
        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.fc2(x)
        return x

# 모델 학습 함수
def train(model, trainloader, optimizer, criterion, log_interval, epoch):
    model.train()  # 모델을 학습 모드로 설정
    for batch_idx, (image, label) in enumerate(trainloader):
        image = image.to(DEVICE)  # 이미지를 기기로 이동
        label = label.to(DEVICE)  # 레이블을 기기로 이동
        optimizer.zero_grad()  # 그래디언트 초기화
        output = model(image)  # 모델에 이미지 전달하여 예측
        loss = criterion(output, label)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트

        # 현재 진행 중인 학습 상황 출력
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(image), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
    
# 모델 평가 함수
def evaluate(model, testloader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in testloader:
            image = image.to(DEVICE)  # 이미지를 기기로 이동
            label = label.to(DEVICE)  # 레이블을 기기로 이동
            output = model(image)  # 모델에 이미지 전달하여 예측
            test_loss += criterion(output, label).item()  # 손실 누적
            prediction = output.max(1, keepdim=True)[1]  # 가장 높은 확률을 가진 클래스 선택
            correct += prediction.eq(label.view_as(prediction)).sum().item()  # 정확한 예측 수 누적
    
    # 손실 평균 및 정확도 계산
    test_loss /= (len(testloader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, test_accuracy

# 메인 함수
def main():
    # 학습 및 테스트 데이터셋 불러오기
    # trainset, testset의 root를 자신의 폴더에 맞게 변경해주세요.
    trainloader, testloader = prepare_data(
        root_train="/home/zxro/DeepLearning/dataset",
        root_test="/home/zxro/DeepLearning/testset",
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    model = CNN().to(DEVICE)  # 모델 초기화 및 기기 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()  # 손실 함수 정의
    print(model)

    # 학습 및 평가 반복
    for epoch in range(1, EPOCHS + 1):
        train(model, trainloader, optimizer, criterion, log_interval=200, epoch=epoch)
        test_loss, test_accuracy = evaluate(model, testloader, criterion)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
            epoch, test_loss, test_accuracy))

    torch.save(model, 'modelcnn.pt')  # 학습된 모델 저장

if __name__ == "__main__":
    main()
