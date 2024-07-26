import torch
import torch.nn as nn
from torch.nn import ModuleList

class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels, prune_rate = 1.0):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, int(6 * prune_rate), kernel_size=5, stride=1, padding=2, bias = False)
        self.bn1 = nn.BatchNorm2d(int(6 * prune_rate))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(int(6 * prune_rate), 16, kernel_size=5, stride=1, padding=0, bias = False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

        self.name = "LeNet5"

        
    def forward(self, x):
        x = 2.0 * x - 1.0

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        
        return x