from torch import nn
from torch.nn import functional as F
import torch
from operator import mul as multiplicator
from functools import reduce


class BaseNet(nn.Module):
    def __init__(self, chan1=32, chan2=64, chan3=128, nb_hidden1=50, nb_hidden2=50, nb_hidden3=25):
        super(BaseNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv2d(2, chan1, kernel_size=3, groups=2))
        self.features.add_module("relu_1", nn.ReLU())
        self.features.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv_2", nn.Conv2d(chan1, chan2, kernel_size=3, padding=1))
        self.features.add_module("relu_2", nn.ReLU())
        self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=2))
        self.features.add_module("conv_3", nn.Conv2d(chan2, chan3, kernel_size=2))
        self.features.add_module("relu_3", nn.ReLU())

        class_size = self.features(torch.empty((1, 2, 14, 14))).shape
        self.linear_size = reduce(multiplicator, list(class_size[1:]))

        self.fc1 = nn.Linear(self.linear_size, nb_hidden1)
        self.fc2 = nn.Linear(nb_hidden1, nb_hidden2)
        self.fc3 = nn.Linear(nb_hidden2, nb_hidden3)
        self.fc4 = nn.Linear(nb_hidden3, 2)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc1(x.view(-1, self.linear_size)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
