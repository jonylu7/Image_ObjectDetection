import torch
from Model.YOLO import YOLO
from loss import lossFunction


def main():
    yolov1=YOLO()
    yolov1.eval()
    print(yolov1)
    test_data=torch.zeros([3,448,448])
    result=yolov1(test_data)

    print(result.shape)
    loss = lossFunction()
    loss(result, result)


main()