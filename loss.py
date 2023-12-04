import torch
from torch import nn


class lossFunction(nn.Module):
    def __init__(self,lambda_coord:int,lambda_noobj:float):
        super().__init__()
        self.lambda_coord=lambda_coord
        self.lambda_noobj=lambda_noobj




    def forward(self,y:torch.Tensor,y_pred:torch.Tensor):
        return 0
