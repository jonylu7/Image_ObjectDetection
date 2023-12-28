import torch
from torch import nn
import config
from utils import get_iou


class lossFunction(nn.Module):
    def __init__(self,lambda_coord:int,lambda_noobj:float):
        super().__init__()
        self.lambda_coord=lambda_coord
        self.lambda_noobj=lambda_noobj
        self.S=config.S
        self.B=config.B
        self.C=config.C

    def fromTensorToCoord(self,data:torch.tensor,row,col):
        x=0
        y=0
        width=0
        height=0

        grid_size_x = config.IMAGE_SIZE[0] / config.S
        grid_size_y = config.IMAGE_SIZE[1] / config.S


        x=data[config.C]*1
        y=data[config.C+1]
        width=data[config.C+2]
        height=data[config.C+3]


        return (x,y),(x+width,y+height)
    def calcualteC(self,y_pred_grid:torch.Tensor,y_gird:torch.Tensor,col,row):
        ##1. transfer to normal coord
        ##2. find intersection coord and calc size
        ##3. calc total size- intersection size
        ##4. intersection/ total

        intersection=0
        total=0




        return 0


    def calculateEachGrid(self,y_pred_grid:torch.Tensor,y_gird:torch.Tensor,col,row):
        sum=0
        for c in range(config.C):
            sum+=pow(y_gird[:c]-y_pred_grid[:c],2)+self.lambda_noobj*pow(y_gird[:c]-y_pred_grid[:c],2)

        ## xywh
        xywh=0
        for i in range(10):
            xywh+=pow(y_gird[config.C+i]-y_pred_grid[config.C+i],2)
        xywh*=self.lambda_coord
        sum+=xywh
        sum+=self.calculateC(y_pred_grid,y_gird,col,row)

        return sum


    def forward(self,y_pred:torch.Tensor,y:torch.Tensor):
        loss=0
        for row in range(self.S):
            for col in range(self.S):
                loss+=self.calculateEachGrid(y_pred[row,col,:],y[row,col,:],col,row)

        return loss



