from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, SpikeDataset, load_model, CatNet, fuse_module
import numpy as np
from sklearn.metrics import confusion_matrix
import catCuda

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, 1.0)
    return out
class ECGDataset(torch.utils.data.Dataset):
    # type - random, spike, float
    def __init__(self, path, train=False):
        self.data = np.load(path)
        if train:
            self.x = self.data['x_train']
            self.y = self.data['y_train']
        else:
            self.x = self.data['x_test']
            self.y = self.data['y_test']
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index].reshape((180,1,1))), self.y[index]
    def __len__(self):
        return len(self.y)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3,1), 1,(1,0), bias=True)
        self.Bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 24, (3,1), 1,(1,0), bias=True)
        self.Bn2 = nn.BatchNorm2d(24)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(24*45, 2, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=0, max=1)
        x = self.dropout1(x)
        x = torch.div(torch.ceil(torch.mul(x,10)),10)
        x = F.avg_pool2d(x, (2,1))
        x = torch.div(torch.ceil(torch.mul(x,10)),10)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = torch.clamp(x, min=0, max=1)
        x = torch.div(torch.ceil(torch.mul(x,10)),10)
        x = self.dropout2(x)

        x = F.avg_pool2d(x, (2,1))
        x = torch.div(torch.ceil(torch.mul(x,10)),10)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.clamp(x, min=0, max=1)
        x = self.dropout3(x)


        return x


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn


        self.conv1 = snn.conv(1, 16, (3,1), 1, (1,0),bias=True)
        self.conv2 = snn.conv(16, 24, (3,1), 1, (1,0),bias=True)
        self.pool1 = snn.pool((2,1))
        self.pool2 = snn.pool((2,1))

        self.fc1 = snn.dense((1,45,24), 2, bias=True)

    def forward(self, x):
        x = self.snn.spike(self.conv1(x))
        x = self.snn.spike(self.pool1(x))
        x = self.snn.spike(self.conv2(x))
        x = self.snn.spike(self.pool2(x))
        x = self.fc1(x)

        return self.snn.sum_spikes(x)/self.T

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pre_l = []
    t = []
    pre_l_ = []
    pre_l_1 = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor) 
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pre_l.append(pred)
            t.append(target)
            #print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    #print()
    test_loss /= len(test_loader.dataset)
    pre_l = np.array(pre_l)
    #print(pre_l,t)
    t = np.array(t)
    
    
    for item in pre_l[0]:
        pre_l_.append(item[0].cpu().numpy())
    for i in range(len(pre_l_)):
        pre_l_[i] = int(pre_l_[i])
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct


def test_conf_s(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pre_l = []
    t = []
    pre_l_ = np.ones(1832)
    pre_l_1 = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor) 
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for item in pred:
                pre_l.append(int(item))
            t.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    pre_l = np.array(pre_l)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #print(pre_l)
    return pre_l

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=40, metavar='N',
                        help='SNN time window')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    path = './ecg_data_normalize_smoke_2c.npz'  
    #path = './all_0810_paper_smoke_normalize_01.npz'
    f = np.load(path)
    train_x, train_y = f['x_train'], f['y_train']
    test_x, test_y = f['x_test'], f['y_test']
    """
    for i in range(len(train_x)):
        train_x[i] = minmaxscaler (train_x[i])

    for i in range(len(test_x)):
        test_x[i] = minmaxscaler (test_x[i])
    """


    y_test_ = test_y
    X_train_ = torch.FloatTensor(train_x)
    print(X_train_.shape)


    X_train_ = X_train_.reshape(-1,1,180,1)
    X_train_ = torch.clamp(X_train_, min=0, max=1)
    X_train_ = torch.div(torch.ceil(torch.mul(X_train_,8)),8)
    y_train_ = torch.FloatTensor(train_y)
    
    X_test = torch.FloatTensor(test_x)
    X_test = X_test.reshape(-1,1,180,1)
    X_test = torch.clamp(X_test, min=0, max=1)
    X_test = torch.div(torch.ceil(torch.mul(X_test,4)),4)

    y_test = torch.FloatTensor(test_y)

    

    torch_dataset_train = torch.utils.data.TensorDataset(X_train_, y_train_)
    torch_dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
    snn_dataset = SpikeDataset(torch_dataset_test, T = args.T)

    train_loader = torch.utils.data.DataLoader(torch_dataset_train,shuffle=True,batch_size=256*3)
    test_loader = torch.utils.data.DataLoader(torch_dataset_test, shuffle=False,batch_size=64)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, shuffle=False,batch_size=1)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)

    if args.resume != None:
        load_model(torch.load(args.resume), model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    Acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader)
        Acc_ = test(model, device, test_loader)
        if Acc_>Acc:
            Acc = Acc_
            torch.save(model.state_dict(), "ecg2c_1.pt")
        scheduler.step()
    
    fuse_module(model)
    transfer_model(model, snn_model)
    test(snn_model, device, snn_loader)


if __name__ == '__main__':
    main()
