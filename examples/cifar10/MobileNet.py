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
from catSNN import spikeLayer, transfer_model, SpikeDataset ,load_model, fuse_module, transfer_model_sequence,fuse_bn_recursively
import catSNN
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = self._make_layers()
        self.fc = nn.Linear(512, 10, bias=True)
        self.model.apply(initialize_weights)


    def forward(self, x):
        out = self.model(x)
        out = out.view(-1,512)
        out = self.fc(out)
        return out

    def _make_layers(self):
        layers = []
        cfg = [(64,1), (128,1), (256,1) ,(256,2), (256,1), (512,1),(512,2),'M']

        layers += [nn.Conv2d(3, 32, 3, 1, 1, bias=True), catSNN.Clamp(max = 1)]
        in_channels = 32
        for x in cfg:
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=8, stride=8)]
            else:
                layers += [nn.Conv2d(in_channels,2*in_channels,1,1,0,bias = True),catSNN.Clamp(max = 1)]
                layers += [nn.Conv2d(2*in_channels,2*in_channels,3,x[1],1,bias=True, groups=2*in_channels),catSNN.Clamp(max = 1)]
                layers += [nn.Conv2d(2*in_channels,x[0],1,1,0,bias = True),catSNN.Clamp(max = 1)]
                in_channels = x[0]
        return nn.Sequential(*layers)



class CatNet(nn.Module):
    def __init__(self,  T):
        super(CatNet, self).__init__()
        self.snn = catSNN.spikeLayer(T)
        self.T = T
        self.model = self._make_layers()
        self.fc = self.snn.dense(512,10,bias = True)
  

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        out = self.snn.sum_spikes(out)/self.T
        return out

    def _make_layers(self):
        layers = []
        cfg = [(64,1), (128,1), (256,1) ,(256,2), (256,1), (512,1),(512,2),'M']
        layers += [self.snn.conv(3, 32, 3,1,1 ,bias=True),self.snn.spikeLayer()]
        in_channels = 32
        for x in cfg:
            if x == 'M':
                layers += [self.snn.pool(8)]
            else:
                layers += [self.snn.conv(in_channels,2*in_channels,1,1,0,bias = True),
                           self.snn.spikeLayer()]
                layers += [self.snn.conv(2*in_channels,2*in_channels,3,x[1],1,bias=True, groups=2*in_channels),
                           self.snn.spikeLayer()]
                layers += [self.snn.conv(2*in_channels,x[0],1,1,0,bias = True),
                           self.snn.spikeLayer()]
                in_channels = x[0]
        # layers += [self.snn.sum_spikes_layer()]
        return nn.Sequential(*layers)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #onehot = torch.nn.functional.one_hot(target, 10)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
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
    parser.add_argument('--T', type=int, default=500, metavar='N',
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

    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(mean, std)
        ])


    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform_train)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform_test)
    snn_dataset = SpikeDataset(dataset2, T = args.T)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=1, shuffle=False)

    model = Net().to(device)
    snn_model = CatNet(args.T).to(device)

    if args.resume != None:
        load_model(torch.load(args.resume), model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, train_loader)
        test(model, device, test_loader)
        scheduler.step()
    #model = fuse_bn_recursively(model)
    transfer_model(model, snn_model)
    test(snn_model, device, snn_loader)

    #if args.save_model:
    


if __name__ == '__main__':
    main()
