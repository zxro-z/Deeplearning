import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from ox_crop import imgcut, get_image_files
from ox_cnn import CNN  # ox_cnn 모듈이 있어야 함

class CustomDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def main():
    # 이미지 증강
    trans = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 모델과 이미지 파일 경로 초기화
    imgpath = "/home/zxro/DeepLearning/otherdata/1000011917.jpg"
    imgcut(imgpath)
    inferdir = "/home/zxro/DeepLearning/testset_crop"

    # 이미지 파일들을 불러와 데이터셋 생성
    image_files = get_image_files(inferdir)
    infer_dataset = CustomDataset(image_files, transform=trans)

    # 미리 학습된 모델을 불러와서 예측 수행
    model = torch.load("modelcnn.pt", map_location=torch.device('cpu'))
    model.eval()

    predictions = []

    # 각 이미지에 대해 모델을 통해 예측 수행
    for idx in range(len(infer_dataset)):
        img = infer_dataset[idx]
        out = model(img.unsqueeze(0))
        _, predict = torch.max(out.data, 1)
        predictions.append(predict.item())

    print("Predictions:", predictions)

    figure = plt.figure(figsize=(10, 5))
    cols, rows = 5, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(infer_dataset), size=(1,)).item()
        img = infer_dataset[sample_idx]
        img = img.unsqueeze(0)  
        figure.add_subplot(rows, cols, i)
        out = model(img.cpu())  
        _, predict = torch.max(out.data, 1)
        img = img.squeeze() 
        img = img.numpy()  
        img = np.transpose(img, (1, 2, 0))  
        plt.title("actual: {}".format(predict.item()))
        plt.axis("off")
        plt.imshow(img)
    plt.savefig('output.png', facecolor='w')
    plt.show()

if __name__ == '__main__':
    main()
