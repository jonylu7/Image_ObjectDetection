import torch

import config
from Model.YOLO import YOLO,YOLO_default
from loss import lossFunction
from data import readData
from torchvision import transforms
import config


def main():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(config.IMAGE_SIZE)])
    train_data_loader, test_data_loader = readData(transform)

    yolov1=YOLO_default()
    yolov1.eval()
    result=yolov1(next(iter(train_data_loader))[0])

    print(result.shape)
    #loss = lossFunction()
    #loss(result, result)


main()