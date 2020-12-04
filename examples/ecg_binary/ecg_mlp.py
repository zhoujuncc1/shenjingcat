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
from catSNN import spikeLayer, transfer_model, SpikeDataset, load_model, CatNet
import numpy as np

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
        self.fc1 = nn.Linear(180, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, 2, bias=False)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.relu(x)
        return output

'''
class CatNet(nn.Module):

    def __init__(self, T):
        super(CatNet, self).__init__()
        self.T = T
        snn = spikeLayer(T)
        self.snn=snn
        self.conv1 = snn.conv(1, 12, (1,3), padding=(0,1))
        self.fc1 = snn.dense((90,1,12), 512)
        self.fc2 = snn.dense(512, 256)
        self.fc3 = snn.dense(256, 2)

        self.pool1 = snn.pool(2)

    def forward(self, x):
        x = self.snn.spike(self.conv1(x))
        x = self.pool1(x)
        x = self.snn.spike(self.fc1(x))
        x = self.snn.spike(self.fc2(x))
        x = self.snn.spike(self.fc3(x))

        return self.snn.sum_spikes(x)/self.T
'''
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 2)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, onehot.type(torch.float))
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 2)
            output = model(data)
            test_loss += F.mse_loss(output, onehot.type(torch.float), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
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
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=60, metavar='N',
                        help='SNN time window')
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

    dataset1 = ECGDataset('examples/ecg_binary/data.npz', train=True)
    dataset2 = ECGDataset('examples/ecg_binary/data.npz', train=False)
    snn_dataset = SpikeDataset(dataset2, T = args.T)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, **kwargs)

    model = Net()
    snn_model = CatNet(model, args.T, (180))
    model = model.to(device)
    snn_model = snn_model.to(device)
    if args.resume != None:
        #load_model(torch.load(args.resume), snn_model)
        model.load_state_dict(torch.load(args.resume), strict=False)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        #train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        
        snn_model.load_weight(model)
        test(snn_model, device, snn_loader)
        
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "ecg.pt")


if __name__ == '__main__':
    main()