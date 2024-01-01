import torch
from torch import nn
import config
from utils import get_iou


class lossFunction(nn.Module):
    def __init__(self, lambda_coord: int, lambda_noobj: float):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.S = config.S
        self.B = config.B
        self.C = config.C

    def fromTensorToCoord(self, data: torch.tensor):
        width = data[config.C + 2]
        height = data[config.C + 3]

        x1 = data[config.C] - width / 2.0
        x2 = data[config.C] + width / 2.0
        y1 = data[config.C + 1] - height / 2.0
        y2 = data[config.C + 1] + height / 2.0

        return (x1, y1), (x2, y2)

    def getArea(self, data: torch.Tensor):
        width = data[config.C + 2]
        height = data[config.C + 3]
        return width * height

    def calcualteC(self, y_pred_grid: torch.Tensor, y_gird: torch.Tensor, col, row):
        ##1. transfer to normal coord
        ##2. find intersection coord and calc size
        ##3. calc total size- intersection size
        ##4. intersection/ total

        pred_tl, pred_br = self.fromTensorToCoord(y_pred_grid)
        pred_Area = self.getArea(y_pred_grid)
        y_tl, y_br = self.fromTensorToCoord(y_gird)
        y_Area = self.getArea(y_gird)

        tl = (max(y_tl[0], pred_tl[0]), min(y_tl[1], pred_tl[1]))
        br = (max(y_br[0], pred_br[0]), min(y_br[1], pred_br[1]))

        intersectioArea = (br[0] - tl[0]) * (tl[1] - br[1])
        union = pred_Area + y_Area - intersectioArea

        return intersectioArea / union

    def calculateEachGrid(self, y_pred_grid: torch.Tensor, y_gird: torch.Tensor, col, row):
        sum = 0
        for c in range(config.C):
            sum += pow(y_gird[:c] - y_pred_grid[:c], 2) + self.lambda_noobj * pow(y_gird[:c] - y_pred_grid[:c], 2)

        ## xywh
        xywh = 0
        for i in range(10):
            xywh += pow(y_gird[config.C + i] - y_pred_grid[config.C + i], 2)
        xywh *= self.lambda_coord
        sum += xywh
        sum += self.calculateC(y_pred_grid, y_gird, col, row)

        return sum

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        loss = 0
        for row in range(self.S):
            for col in range(self.S):
                loss += self.calculateEachGrid(y_pred[row, col, :], y[row, col, :], col, row)

        return loss
