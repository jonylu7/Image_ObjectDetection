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

def labelResize(label,image_size):
    x_scale=image_size[0]/int(label['size']['width'])
    y_scale=image_size[1]/int(label['size']['height'])

    for obj in label['object']:
        obj['bndbox']['xmin']=str(int(obj['bndbox']['xmin'])*x_scale)
        obj['bndbox']['xmax']=str(int(obj['bndbox']['xmax'])*x_scale)
        obj['bndbox']['ymin']=str(int(obj['bndbox']['ymin'])*y_scale)
        obj['bndbox']['ymax']=str(int(obj['bndbox']['ymax'])*y_scale)


class YOLOLoadVOCDataset(Dataset):
    def __init__(self,train:bool,transofrm,download,year,data_location,S,B,C,image_size):
        assert year in config.VOCYEAR
        self.dataset=VOCDetection(year=year
                                      ,download=download
                                    ,transform=transform
                                      ,root=data_location
                                    ,image_set=('train' if train else 'test')
            )


        self.S = S
        self.B = B
        self.C = C
        self.imageSize = image_size
        self.class_file_location = str(PurePath(
            data_location).parent) + "/VOC" + self.dataset.image_set + "_" + self.dataset.year + ".json"
        self.classWithFileName = self.loadClasses()
        self.classnames=list(self.classWithFileName.keys())



    def loadClasses(self):
        filelocaton=self.class_file_location
        result=loadVOCClasses(filelocaton)
        index=0
        if(len(result)==0):
            for image, label in self.dataset:
                labelResize(label['annotation'],self.imageSize)
                for item in label['annotation']['object']:
                    if(item['name'] not in result.keys()):
                        result[item['name']]=index
                        index+=1
            saveVOCClasses(result,filelocaton)
        return result



    def __getitem__(self, i):
        data,label=self.dataset[i]

        ground_truth=torch.zeros(self.S,self.S,(self.B*5+self.C))
        for rows in range(self.S):
            for cols in range(self.S):
                ground_truth[rows,cols,0:]



        return(label)




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
    DisplayDataLoaderResult_VOC(train_data_loader)
    DisplayDataLoaderResult_VOC(test_data_loader)
    train_data_location=config.DATA_PATH + "og_data/archive/VOCtrainval_06-Nov-2007"
    _,_=YOLOLoadVOCDataset(True,transform,False,'2007',train_data_location,config.S,config.B,config.C,config.IMAGE_SIZE)[0]







