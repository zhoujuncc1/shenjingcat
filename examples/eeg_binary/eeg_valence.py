from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, SpikeDataset ,load_model, fuse_module,normalize_weight
from scipy import signal
from sklearn.metrics import confusion_matrix

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 0,bias=True)
        self.Bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 1,0,bias=True)
        self.Bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, 1, 0,bias=True)
        self.Bn3 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512,128, bias=True)
        self.fc2 = nn.Linear(128, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=0, max=1)
        #x = torch.div(torch.ceil(torch.mul(x,100)),100)

        x = self.conv2(x)
        x = self.Bn2(x)
        x = torch.clamp(x, min=0, max=1)
        #x = torch.div(torch.ceil(torch.mul(x,100)),100)

        x = F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = self.Bn3(x)
        x = torch.clamp(x, min=0, max=1)
        #x = torch.div(torch.ceil(torch.mul(x,100)),100)

        x = F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.clamp(x, min=0, max=1)
        #x = torch.div(torch.ceil(torch.mul(x,100)),100)
        output = self.fc2(x)

        return output


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn

        self.conv1 = snn.conv(1, 16, 5, 1,0,bias=True)
        self.conv2 = snn.conv(16, 32, 5, 1,0, bias=True)
        self.conv3 = snn.conv(32, 32,5,1,0, bias=True)
        self.pool1 = snn.pool(2)
        self.pool2 = snn.pool(2)
        self.fc1 = snn.dense((4,4,32), 128, bias=True)
        self.fc2 = snn.dense(128, 2, bias=True)

    def forward(self, x):
        x = self.snn.spike(self.conv1(x))
        x = self.snn.spike(self.conv2(x))
        x = self.snn.spike(self.pool1(x))
        x = self.snn.spike(self.conv3(x))
        x = self.snn.spike(self.pool2(x))
        x = self.snn.spike(self.fc1(x))
        x = self.snn.spike(self.fc2(x))
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
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    pre_l = np.array(pre_l)
    for item in pre_l[0]:
        pre_l_.append(item[0].cpu().numpy())
    for i in range(len(pre_l_)):
        pre_l_[i] = int(pre_l_[i])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct
def test_conf(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pre_l = []
    t = []
    pre_l_ = []
    pre_l_1 = []
    #all_pre = []
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
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    pre_l = np.array(pre_l)
    for item in pre_l[0]:
        pre_l_.append(item[0].cpu().numpy())
    for i in range(len(pre_l_)):
        pre_l_[i] = int(pre_l_[i])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return pre_l_

def test_conf_s(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pre_l = []
    t = []
    pre_l_ = np.ones(1832)
    pre_l_1 = []
    #all_pre = []
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



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--T', type=int, default=100, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    

    f = np.load('dataeeg/valence.npz')
    X_train_ = f['X_train_']
    y_train_ = f['y_train_']
    X_test = f['X_test']
    
    y_test = f['y_test']
    y_test_ = y_test

    X_train_ = torch.FloatTensor(X_train_)
    y_train_ = torch.FloatTensor(y_train_)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    X_train_ = X_train_*3000 + (0.001)*torch.randn(len(X_train_), len(X_train_[0]), len(X_train_[0][0]),len(X_train_[0][0][0]))
    X_train_ = np.clip(X_train_, 0, 1)
    X_test = X_test*3000
    X_test = np.clip(X_test, 0, 1)
    
    for i in range(4):
        X_train_ = torch.cat([X_train_,X_train_],axis=0)
        y_train_ = torch.cat([y_train_,y_train_],axis=0)
    

    torch_dataset_train = torch.utils.data.TensorDataset(X_train_, y_train_)
    torch_dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
    snn_dataset = SpikeDataset(torch_dataset_test, T = args.T)
    train_loader = torch.utils.data.DataLoader(torch_dataset_train,shuffle=True,batch_size=128)
    test_loader = torch.utils.data.DataLoader(torch_dataset_test, shuffle=False,batch_size=1832)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, shuffle=False,batch_size=1)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)
    #"v0126_.pt"
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
            fuse_module(model)
            torch.save(model.state_dict(), "v0126_3.pt")

        scheduler.step()
    print(Acc)
    pre_ = test_conf(model, device, test_loader)
    true_ = y_test_

    print(confusion_matrix(true_, pre_))

    fuse_module(model)
    transfer_model(model, snn_model)

    for param_tensor in snn_model.state_dict():
        print(param_tensor, "\t", snn_model.state_dict()[param_tensor].size())

    pre_s = test_conf_s(snn_model, device, snn_loader)
    print(confusion_matrix(true_, pre_s))

if __name__ == '__main__':
    main()
