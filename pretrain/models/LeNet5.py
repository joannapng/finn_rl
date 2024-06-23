import torch
import torch.nn as nn
from torch.nn import ModuleList

class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(LeNet5, self).__init__()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2))
        self.conv_features.append(nn.BatchNorm2d(6))
        self.conv_features.append(nn.ReLU())
        self.conv_features.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.conv_features.append(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0))
        self.conv_features.append(nn.BatchNorm2d(16))
        self.conv_features.append(nn.ReLU())
        self.conv_features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.linear_features.append(nn.Linear(400, 120))
        self.linear_features.append(nn.ReLU())
        self.linear_features.append(nn.Linear(120, 84))
        self.linear_features.append(nn.ReLU())
        self.linear_features.append(nn.Linear(84, num_classes))

        self.name = "LeNet5"

    def clip_weights(self, min_val = -1, max_val = 1):
        for mod in self.conv_features:
            if isinstance(mod, nn.Conv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, nn.Linear):
                mod.weight.data.clamp_(min_val, max_val)
        
    def forward(self, x):
        x = 2.0 * x - 1.0

        for mod in self.conv_features:
            x = mod(x)

        x = x.flatten(1)
        for mod in self.linear_features:
            x = mod(x)
        
        return x