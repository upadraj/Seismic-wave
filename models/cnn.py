import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=2, padding=1)    
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=2, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 27 * 27, 512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):        
        x = F.tanh(self.conv1(x))
        x = self.pool(F.tanh((self.conv2(x))))
        x = F.tanh((self.conv3(x)))
        x = x.view(-1, 512 * 27 * 27)
        x = F.tanh(self.fc1(x))
        # x = F.tanh(self.bn_fc1(self.fc1(x)))  
        # x = F.tanh(self.bn_fc1(self.fc1(x)))  
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x