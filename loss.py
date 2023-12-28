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



    def calcualteC(self,y_pred_grid:torch.Tensor,y_gird:torch.Tensor):

        return 0


    def calculateEachGrid(self,y_pred_grid:torch.Tensor,y_gird:torch.Tensor):
        sum=0
        for c in range(config.C):
            sum+=pow(y_gird[:c]-y_pred_grid[:c],2)+self.lambda_noobj*pow(y_gird[:c]-y_pred_grid[:c],2)

        ## xywh
        xywh=0
        for i in range(10):
            xywh+=pow(y_gird[config.C+i]-y_pred_grid[config.C+i],2)
        xywh*=self.lambda_coord
        sum+=xywh
        sum+=self.calculateC(y_pred_grid,y_gird)

        return sum


    def forward(self,y_pred:torch.Tensor,y:torch.Tensor):
        loss=0
        for row in range(self.S):
            for col in range(self.S):
                loss+=self.calculateEachGrid(y_pred[row,col,:],y[row,col,:])

        return loss



