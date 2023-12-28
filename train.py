from engine import train
from Model.YOLO import YOLO
from loss import lossFunction
from data import YOLOLoadVOCDataset,readData
from torchvision import transforms
import config
import torch
from torch.utils.tensorboard import SummaryWriter

def main(train_data,test_data):

    weight_decay=0.0005
    epoches=80


    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        device=torch.device("mps")


    model = YOLO()
    loss_fn = lossFunction()
    learningRate=0
    optimizer=torch.optim.SGD(model.parameters(),lr=learningRate,weight_decay=weight_decay,momentum=0.9)

    writer=SummaryWriter()
    result=train(epoches,model,loss_fn,optimizer,train_data,test_data,device,writer)




if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(config.IMAGE_SIZE)
                                       , transforms.ToTensor()])

    train_data_loader, test_data_loader = readData(transform)
    train_data_location = config.DATA_PATH + "og_data/archive/VOCtrainval_06-Nov-2007"
    train_dataset = YOLOLoadVOCDataset(True, transform, False, '2007', train_data_location, config.S, config.B, config.C,
                                 config.IMAGE_SIZE)
    test_dataset=YOLOLoadVOCDataset(False, transform, False, '2007', train_data_location, config.S, config.B, config.C,
                                 config.IMAGE_SIZE)
    main(train_dataset,test_data_loader)


