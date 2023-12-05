import json
import os
import torch
from torchvision import transforms
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import DataLoader,Dataset
import config
import utils
from utils import DisplayDataLoaderResult_VOC
from pathlib import PurePath


def loadVOCClasses(classfileLocation)->dict:
    try:
        with open(classfileLocation,'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Didn't find the required file")
        return dict()



def saveVOCClasses(classdict,classfileLocation):
    assert PurePath(classfileLocation).suffix == '.json'
    with open(classfileLocation,'w+') as file:
        json.dump(classdict,file,indent=2)




class YOLOLoadVOCDataset(Dataset):
    def __init__(self,train:bool,transofrm,download,year,data_location):
        assert year in config.VOCYEAR
        self.dataset=VOCDetection(year=year
                                      ,download=download
                                    ,transform=transform
                                      ,root=data_location
                                    ,image_set=('train' if train else 'test')
            )

        self.transfrom=transofrm
        self.class_file_location = str(PurePath(
            data_location).parent) + "/VOC" + self.dataset.image_set + "_" + self.dataset.year + ".json"
        self.classWithFileName=self.loadClasses()
        self.classnames=list(self.classWithFileName.keys())

    def loadClasses(self):
        filelocaton=self.class_file_location
        result=dict
        result=loadVOCClasses(filelocaton)
        if(len(result)==0):
            for image, label in self.dataset:
                for item in label['annotation']['object']:
                    if(item['name'] not in result.keys()):
                        result[item['name']]=[label['annotation']['filename']]
                    else:
                        result[item['name']].append(label['annotation']['filename'])
            saveVOCClasses(result,filelocaton)
        return result



    def __getitem__(self, i):
        data,label=self.dataset[i]









        return(data,label)




def loadDataSet(data_location,type_of_data,batch_size,shuffle,num_workers,transform)->DataLoader:
    voc_dataset=VOCDetection(download=False,root=data_location,year='2007',transform=transform,image_set=type_of_data)
    return voc_dataset



def readData(transform):
    train_data_location=config.DATA_PATH+"og_data/archive/VOCtrainval_06-Nov-2007"
    test_data_location = config.DATA_PATH + "og_data/archive/VOCtest_06-Nov-2007"
    train_data_loader=loadDataSet(train_data_location,type_of_data="train",batch_size=config.BATCH_SIZE,shuffle=False,num_workers=os.cpu_count(),transform=transform)
    test_data_loader=loadDataSet(test_data_location,type_of_data="test",batch_size=config.BATCH_SIZE,shuffle=True,num_workers=os.cpu_count(),transform=transform)

    return train_data_loader,test_data_loader




if __name__ == '__main__':
    transform=transforms.Compose([transforms.Resize(config.IMAGE_SIZE)
                                     ,transforms.ToTensor()])

    train_data_loader,test_data_loader=readData(transform)
    #DisplayDataLoaderResult_VOC(train_data_loader)
    #DisplayDataLoaderResult_VOC(test_data_loader)
    train_data_location=config.DATA_PATH + "og_data/archive/VOCtrainval_06-Nov-2007"
    _,_=YOLOLoadVOCDataset(True,transform,False,'2007',train_data_location)[0]







