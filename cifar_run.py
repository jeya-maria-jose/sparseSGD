from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision
import timeit
from optimnew import *

def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img.cpu() + noise
    return noisy_img



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.encoder1 = nn.Conv2d(3, 16, 3, stride = 1,padding=1)  
        self.encoder2=   nn.Conv2d(16, 32, 3, stride  =1,padding=1)  
        self.encoder3 =   nn.Conv2d(32, 64, 3, stride  =1,padding=1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x2 = x
        x = self.encoder1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.encoder2(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = F.relu(self.encoder3(x))
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        noisy_data = add_noise(data)
        noisy_data = noisy_data.cuda()        
        optimizer.zero_grad()
        output = model(noisy_data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
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
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = sparseSGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    timetot = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        start = timeit.default_timer()
        train(args, model, device, train_loader, optimizer, epoch)
        

        test(args, model, device, test_loader)
        scheduler.step()

        stop = timeit.default_timer()
        timetmp = stop - start
        timetot = timetmp + timetot
        print('Time: ', stop - start)  


    if args.save_model:
        torch.save(model.state_dict(), "cifar_cnn.pt")

    print("Total time = ", timetot )


if __name__ == '__main__':
    main()