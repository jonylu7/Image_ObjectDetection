import os
import torch
from torchvision import transforms
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import DataLoader
import config
from utils import DisplayDataLoaderResult_VOC

def loadDataSet(data_location,type_of_data,batch_size,shuffle,num_workers,transform)->DataLoader:
    voc_dataset=VOCDetection(download=False,root=data_location,year='2007',transform=transform,image_set=type_of_data)
    return voc_dataset



if __name__ == '__main__':
    train_data_location=config.DATA_PATH+"og_data/archive/VOCtrainval_06-Nov-2007"
    test_data_location = config.DATA_PATH + "og_data/archive/VOCtest_06-Nov-2007"
    transform=transforms.Compose([transforms.ToTensor()])
    train_data_loader=loadDataSet(train_data_location,type_of_data="train",batch_size=config.BATCH_SIZE,shuffle=False,num_workers=os.cpu_count(),transform=transform)
    test_data_location=loadDataSet(test_data_location,type_of_data="test",batch_size=config.BATCH_SIZE,shuffle=True,num_workers=os.cpu_count(),transform=transform)
    DisplayDataLoaderResult_VOC(train_data_loader)
    DisplayDataLoaderResult_VOC(test_data_location)





