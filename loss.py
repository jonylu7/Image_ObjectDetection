import torch
from torch import nn
from utils import get_iou


class lossFunction(nn.Module):
    def __init__(self,lambda_coord:int,lambda_noobj:float):
        super().__init__()
        self.lambda_coord=lambda_coord
        self.lambda_noobj=lambda_noobj




    def forward(self,y_pred:torch.Tensor,y:torch.Tensor):
        result=get_iou(y_pred,y)
        return 0



