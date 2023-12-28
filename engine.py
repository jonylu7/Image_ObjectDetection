from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_step(model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim,train_data:DataLoader,device:torch.device):

    model.train()
    train_loss,train_acc=0,0
    for batch,(X,y) in enumerate(train_data):
        if batch%10==0:
            print(f"Train_Batch: {batch}")
        X,y=X.to(device),y.to(device)
        y_pred=model(X)

        loss = loss_fn(y_pred,y)
        train_loss+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y == y_pred_class).sum().item() / len(y_pred)

    train_loss = train_loss / len(train_data)
    train_acc = train_acc / len(train_data)

    return train_loss,train_acc


def test_step(model:nn.Module,loss_fn:torch.nn,test_data:DataLoader,device:torch.device):
    model.eval()
    test_loss,test_acc=0,0
    with torch.no_grad():
        for batch,(X,y)in enumerate(test_data):
            if(batch%10==0):
                print(f"Test_Batch: {batch}")
            X,y=X.to(device),y.to(device)
            y_pred_test=model(X)
            test_loss+=loss_fn(y_pred_test,y)
            ## waiting to be implemant

            test_pred_labels = y_pred_test.argmax(dim=1)
            test_acc += ((y == test_pred_labels).sum().item() / len(test_pred_labels))

        test_loss = test_loss / len(test_data)
        test_acc = test_acc / len(test_data)

    return test_loss,test_acc


def train(epoches:int,model:nn.Module,loss_fn:torch.nn,optimizer:torch.optim,train_data:DataLoader,test_data:DataLoader,device:torch.device,writer:SummaryWriter):

    results={
        "train_loss":[],
        "test_loss":[],
        "train_acc":[],
        "test_acc":[]
    }

    for epoch in tqdm(range(epoches)):
        print(f"Epoch:{epoch}:")
        train_loss,train_acc=train_step(model,loss_fn,optimizer,train_data,device)
        test_loss,test_acc=test_step(model,loss_fn,test_data,device)

        print(f"Train_loss:{train_loss:.4f} | Test_loss:{test_loss:.4f} | Train_acc:{train_acc:.4f} | Test_acc:{test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_acc)

        if writer:
            writer.add_scalar(tag="Train/Loss",scalar_value=train_loss,global_step=epoch)
            writer.add_scalar(tag="Train/Acc", scalar_value=train_acc, global_step=epoch)
            writer.add_scalar(tag="Test/Loss", scalar_value=test_loss, global_step=epoch)
            writer.add_scalar(tag="Test/Acc", scalar_value=test_acc, global_step=epoch)
            writer.close()

    return results


