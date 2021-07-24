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
from catSNN import spikeLayer, transfer_model, SpikeDataset ,load_model, fuse_module


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1,0, bias=True)
        self.Bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1,0, bias=True)
        self.Bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.4)
        self.conv3 = nn.Conv2d(32, 32, 4, 2,1, bias=True)
        self.Bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, 1,0, bias=True)
        self.Bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, 1,0, bias=True)
        self.Bn5 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.4)
        self.conv6 = nn.Conv2d(64, 64, 4, 2,1, bias=True)
        self.Bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, 3, 1,0, bias=True)
        self.Bn7 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*4, 10, bias=True)
        self.dropout3 = nn.Dropout2d(0.3)
        #self.fc2 = nn.Linear(128, 10, bias=True)




    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        x = self.conv2(x)
        x = self.Bn2(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        x = self.conv3(x)
        x = self.Bn3(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)
        #x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.Bn4(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        x = self.conv5(x)
        x = self.Bn5(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        #x = self.dropout3(x)

        x = self.conv6(x)
        x = self.Bn6(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        #x = self.dropout2(x)

        x = self.conv7(x)
        x = self.Bn7(x)
        x = torch.clamp(x, min=0, max=1)
        #Please add Q function during retraining
        #x = torch.div(torch.ceil(torch.mul(x,10)),10)

        
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x


class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn

        self.conv1 = snn.conv(1, 32, 3, 1,0,bias=True)
        self.conv2 = snn.conv(32, 32, 3, 1,0,bias=True)
        self.conv3 = snn.conv(32, 32, 4,2,1,bias=True)

        self.conv4 = snn.conv(32, 64, 3, 1,0,bias=True)
        self.conv5 = snn.conv(64, 64, 3, 1,0,bias=True)
        self.conv6 = snn.conv(64, 64, 4, 2,1,bias=True)

        self.conv7 = snn.conv(64, 128, 3, 1,0,bias=True)
     
        self.fc1 = snn.dense((2,2,128), 10, bias=True)
        #self.fc2 = snn.dense(128, 10, bias=True)


    def forward(self, x):
        x = self.snn.spike(self.conv1(x))
        x = self.snn.spike(self.conv2(x))
        x = self.snn.spike(self.conv3(x))
        x = self.snn.spike(self.conv4(x))
        x = self.snn.spike(self.conv5(x))
        x = self.snn.spike(self.conv6(x))
        x = self.snn.spike(self.conv7(x))
        x = self.fc1(x)
        return self.snn.sum_spikes(x)/self.T

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, onehot.type(torch.float))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--T', type=int, default=450, metavar='N',
                        help='SNN time window')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
                        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01)
        ])

    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train)
    
    for i in range(30):
        transform_train_1 = transforms.Compose([

            transforms.RandomRotation(10),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01)
        ])
        dataset1 = dataset1+ datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train_1)
    
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    snn_dataset = SpikeDataset(dataset2, T = args.T)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, **kwargs)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)

    if args.resume != None:
        load_model(torch.load(args.resume), model)
    for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    ACC = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        ACC_ = test(model, device, test_loader)
        if ACC_>ACC or ACC_ == ACC:
            ACC = ACC_
            torch.save(model.state_dict(), "mnist_pretrained.pt")
        
        scheduler.step()
    # After retraining with Q function, you can transfer ANN to SNN.    
    fuse_module(model)
    transfer_model(model, snn_model)
    test(snn_model, device, snn_loader)

    #if args.save_model:



if __name__ == '__main__':
    main()
