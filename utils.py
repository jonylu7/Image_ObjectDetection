import pickle
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Dict
import random
import json
import config

COLOR=['r','g','b','c','m','y']

def unpickle(file):
    with open(file,"rb") as f:
        dict=pickle.load(f,encoding="bytes")

        return dict


def save_model(model:torch.nn.Module,target_dir:str,model_name:str):
    """save trained model in file at the given target dir"""
    target_dir_path=Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_pth=target_dir_path/model_name

    print(f"Save {model_name} at {target_dir_path}")
    torch.save(model.state_dict(),f=model_save_pth)

def save_results(results:Dict,targ_dict:str,fileName):
    """save training result that is stored as dict in file at the given target dict"""
    targ_dict=Path(targ_dict)
    targ_dict.mkdir(parents=True,exist_ok=True)

    with open(targ_dict/fileName,"wb") as file:
        pickle.dump(results,file)
        print(f"dictionary saved to {Path(targ_dict)/fileName}")

def read_results(targ_dict:str,fileName:str):
    """read training result that is stored as dict in file at the given target dict"""
    file_path=Path(targ_dict)/fileName
    with open(file_path,"rb") as file:
        results=pickle.load(file)
        print("dictionary loaded")
    return results


def display_results(results:Dict):
    """display trainloss, trainacc... result by given dict """
    width=range(len(results["train_loss"]))
    plt.figure(figsize=(15,7))
    plt.subplot(2,1,1)
    plt.plot(width,list(map(float,results["train_loss"])),label="train_loss")
    plt.plot(width,list(map(float,results["test_loss"])),label="test_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(width,list(map(float,results["train_acc"])),label="train_acc")
    plt.plot(width,list(map(float,results["test_acc"])),label="test_acc")
    plt.title("Acc")
    plt.legend()

    plt.show()


def transferLabel(label_item):
    """transfer label from str and return xmax,ymax...etc """
    xmin = float(label_item['bndbox']['xmin'])
    ymin = float(label_item['bndbox']['ymin'])
    xmax=float(label_item['bndbox']['xmax'])
    ymax=float(label_item['bndbox']['ymax'])
    width = xmax-xmin
    height = ymax-ymin
    name=label_item['name']
    return xmax,xmin,ymax,ymin,width,height,name

def drawRectangle(label_item,color):
    """draw rectangle of the bounding box"""
    _,x,_,y,width,height,name=transferLabel(label_item)

    rect=patches.Rectangle((x,y) ,width,height,linewidth=2,edgecolor=color,facecolor='none')
    plt.text(x+5,y+20,name,fontsize=12,color=color)
    return rect


def randomChooseColor():
    """"Choose color randomly by predefine COLOR array """
    randNum=random.randint(0,len(COLOR)-1)
    return COLOR[randNum]


def DisplayDataLoaderResult_VOC(dataLoader:DataLoader):
    """"test dataloader is loaded by showing first image through ShowImage function"""
    image,label=next(iter(dataLoader))
    ShowImage(image,label)


def ShowImage(image,label):
    """Show Image and the object box and label name"""
    image = image.permute(1, 2, 0)
    fig, ax = plt.subplots()
    ax.imshow(image)
    for item in label['annotation']['object']:
        color = randomChooseColor()
        rect = drawRectangle(item, color)
        ax.add_patch(rect)
    plt.show()







