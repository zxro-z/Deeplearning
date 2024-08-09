import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
from ox_crop import imgcut,get_image_files
from ox_cnn import CNN 

class CustomDataset(Dataset):

    def __init__(self, image_files, transform=None):
        # 이미지 파일들을 정렬하여 저장합니다.
        self.image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.transform = transform

    def __len__(self):
        # 데이터셋의 총 샘플 수를 반환합니다.
        return len(self.image_files)

    def __getitem__(self, idx):
        # 주어진 인덱스에 해당하는 샘플을 반환합니다.
        img_path = self.image_files[idx]
        # 이미지를 RGB 형식으로 열어서 반환합니다.
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # 이미지에 전처리(transform)를 적용합니다.
            image = self.transform(image)
        return image

def main():
    # 이미지 증강
    trans = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전합니다.
    transforms.RandomRotation(15),  # 이미지를 최대 15도까지 랜덤하게 회전시킵니다.
    transforms.ToTensor(), # 배열 -> Tensor로 변환합니다.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 픽셀 값의 범위를 0~1로 조절합니다.
    transforms.Grayscale(num_output_channels=1)   # grayscale로 변환한다.
    ])

    # 모델과 이미지 파일 경로를 초기화합니다.
    imgpath ="/home/zxro/DeepLearning/otherdata/1000011917.jpg"
    imgcut(imgpath) # 주석 추가: 이미지를 잘라내는 함수 호출
    inferdir = "/home/zxro/DeepLearning/testset_crop"

    # 이미지 파일들을 불러와 데이터셋을 생성합니다.
    image_files = get_image_files(inferdir)
    infer_dataset = CustomDataset(image_files, transform=trans)

    # 미리 학습된 모델을 불러와서 예측을 수행합니다.
    model = torch.load("modelcnn.pt", map_location=torch.device('cpu'))
    model.eval()  
    
    predictions = []
    
    # 각 이미지에 대해 모델을 통해 예측을 수행합니다.
    for idx in range(1, 10):  # 파일 이름이 1부터 9까지 있으므로 반복 범위를 1부터 9까지로 설정합니다.
        img = infer_dataset[idx - 1]  # 파일 이름이 1부터 시작하므로 인덱스를 1씩 줄입니다.
        out = model(img.unsqueeze(0))
        _, predict = torch.max(out.data, 1)
        predictions.append(predict.item())
        
    print("Predictions:", predictions)
    
    #img = imread('lena.png')   # 이미지 읽어오기 (적절한 경로 설정)


if __name__ == '__main__':
    main()