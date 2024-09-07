import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x

        out = F.tanh(self.conv1(x))
        out = self.conv2(out)

        if identity.shape != out.shape:
            identity = self.projection(identity)

        out += identity
        out = F.tanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=1)

        self.resblock1 = ResBlock(32, 64, kernel_size=5, padding=2)
        self.resblock2 = ResBlock(64, 128, kernel_size=7, padding=3)
        self.resblock3 = ResBlock(128, 256, kernel_size=3, padding=1)
        self.resblock4 = ResBlock(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pool(x)

        x = self.resblock2(x)
        x = self.pool(x)

        x = self.resblock3(x)
        x = self.pool(x)

        x = self.resblock4(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 1 * 1)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = self.fc5(x)

        return x
