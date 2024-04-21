import torch
import torch.nn as nn
from torch.nn import ModuleList

class Simple(nn.Module):
	def __init__(self, num_classes, in_channels):
		super(Simple, self).__init__()
		
		self.conv_features = ModuleList()
		self.linear_features = ModuleList()

		self.conv_features.append(nn.Conv2d(1, 1, 5, padding = 2, bias = False))
		self.conv_features.append(nn.ReLU())
		self.conv_features.append(nn.MaxPool2d(kernel_size = 4, stride = 4))

		self.linear_features.append(nn.Linear(7 * 7 , num_classes, bias = False))
		self.name = "Simple"

	def forward(self, x):
		#print("input = " + str(x))
		x = 2.0 * x - 1.0
		#print("after transforming to (-1, 1)" + str(x))

		for mod in self.conv_features:
			x = mod(x)
		#	print(x)

		x = x.flatten(1)

		for mod in self.linear_features:
			x = mod(x)
		#	print(x)
		
		return x