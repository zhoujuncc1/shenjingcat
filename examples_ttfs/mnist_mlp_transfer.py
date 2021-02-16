
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
import torch.autograd.profiler as profiler
import numpy as np

def get_onehot(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class Net(torch.nn.Module):
    def __init__(self, T):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 400, bias=False)
        self.fc2 = torch.nn.Linear(400, 10, bias=False)

    def forward(self, x):
        x = torch.clip(self.fc1(x), 0, 1)
        x = torch.clip(self.fc2(x), 0, 1)
        return x

    def profile(self, x):
        results = []
        x = torch.clip(self.fc1(x), 0, 1)
        results.append(x)
        x = torch.clip(self.fc2(x), 0, 1)
        results.append(x)
        return results

def train(args, model, device, train_loader, optimizer, epoch):
    
    model.train()
    total_loss = torch.tensor(0, device=device, dtype=torch.float32)
    correct = torch.tensor(0, device=device, dtype=torch.float32)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = get_onehot(target, 10).to(device)
        output = model(data.flatten(start_dim=1))
        loss = F.mse_loss(output, onehot)
        total_loss += loss.sum()
        pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.4f}%'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), total_loss/(batch_idx+1), 100. * correct/(batch_idx+1)/len(data)), end='\r')
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = torch.tensor(0, device=device, dtype=torch.float32)
    correct = torch.tensor(0, device=device, dtype=torch.float32)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = get_onehot(target, 10).to(device)
            output = model(data.flatten(start_dim=1))
            loss = F.mse_loss(output, onehot)
            test_loss += loss.sum()
            pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

def profile(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = get_onehot(target, 10).to(device)
            results = model.profile(data.flatten(start_dim=1))
            break
    return results
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
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
    parser.add_argument('--T', type=int, default=256, metavar='N',
                        help='SNN time window')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 =  datasets.MNIST('../data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net(args.T).to(device)

    if args.resume != None:
        model.load_state_dict(torch.load(args.resume), strict=False)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "mnist_mlp_ttfs2_transfer.pt")
    
    profile(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_mlp_ttfs2_transfer_final.pt")


if __name__ == '__main__':
    main()