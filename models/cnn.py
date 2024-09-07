import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=2, padding=1
        )
        # self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=2, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))  # Conv1 -> -> tanh -> Pool
        x = self.pool(F.tanh((self.conv2(x))))  # Conv2 -> BN -> tanh -> Pool
        x = self.pool(F.tanh((self.conv3(x))))  # Conv3 -> BN -> tanh -> Pool
        x = x.view(-1, 256 * 7 * 7)
        x = F.tanh(self.bn_fc1(self.fc1(x)))  # FC1 -> BN -> tanh
        x = F.tanh(self.bn_fc2(self.fc2(x)))  # FC2 -> BN -> tanh
        x = self.fc3(x)
        return x