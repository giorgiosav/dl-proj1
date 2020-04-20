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


class SiameseNet(nn.Module):
    def __init__(self, chan1=16, chan2=32, chan3=64, nb_hidden1=100, nb_hidden2=50, nb_hidden3=50, nb_hidden4=10):
        super(SiameseNet, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("conv_1", nn.Conv2d(1, chan1, kernel_size=3))
        self.features.add_module("relu_1", nn.ReLU())
        self.features.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2))
        self.features.add_module("conv_2", nn.Conv2d(chan1, chan2, kernel_size=2))
        self.features.add_module("relu_2", nn.ReLU())
        self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, dilation=1))
        self.features.add_module("conv_3", nn.Conv2d(chan2, chan3, kernel_size=2))
        self.features.add_module("relu_3", nn.ReLU())

        class_size = self.features(torch.empty((1, 1, 14, 14))).shape
        self.linear_size = reduce(multiplicator, list(class_size[1:]))

        self.classifier1 = nn.Sequential()
        self.classifier1.add_module("linear_1", nn.Linear(self.linear_size, nb_hidden1))
        self.classifier1.add_module("bn_1", nn.BatchNorm1d(nb_hidden1))
        self.classifier1.add_module("relu_1", nn.ReLU())
        self.classifier1.add_module("dropout_1", nn.Dropout(0.25))
        self.classifier1.add_module("linear_2", nn.Linear(nb_hidden1, nb_hidden2))
        self.classifier1.add_module("bn_2", nn.BatchNorm1d(nb_hidden2))
        self.classifier1.add_module("relu_2", nn.ReLU())
        self.classifier1.add_module("dropout_2", nn.Dropout(0.25))
        self.classifier1.add_module("linear_3", nn.Linear(nb_hidden2, 10))

        self.classifier2 = nn.Sequential()
        self.classifier2.add_module("linear_1", nn.Linear(20, nb_hidden3))
        self.classifier2.add_module("bn_1", nn.BatchNorm1d(nb_hidden3))
        self.classifier2.add_module("relu_1", nn.ReLU())
        self.classifier2.add_module("dropout_1", nn.Dropout(0.25))
        self.classifier2.add_module("linear_2", nn.Linear(nb_hidden3, nb_hidden4))
        self.classifier2.add_module("bn_2", nn.BatchNorm1d(nb_hidden4))
        self.classifier2.add_module("relu_2", nn.ReLU())
        self.classifier2.add_module("dropout_2", nn.Dropout(0.25))
        self.classifier2.add_module("linear_3", nn.Linear(nb_hidden4, 2))

    def forward(self, x):
        out_aux = []
        for i in range(0, 2):
            x_i = x[:, i, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
            x_i = self.features(x_i)
            out_aux.append(self.classifier1(x_i.view(-1, self.linear_size)))
        diff = torch.cat((out_aux[1], out_aux[0]), 1)
        out = self.classifier2(diff)
        return out, out_aux


class NonSiameseNet(nn.Module):
    def __init__(self, chan1=16, chan2=32, chan3=64, nb_hidden1=100, nb_hidden2=50, nb_hidden3=50, nb_hidden4=10):
        super(NonSiameseNet, self).__init__()

        self.features1 = nn.Sequential()
        self.features1.add_module("conv_1", nn.Conv2d(1, chan1, kernel_size=3))
        self.features1.add_module("relu_1", nn.ReLU())
        self.features1.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2))
        self.features1.add_module("conv_2", nn.Conv2d(chan1, chan2, kernel_size=2))
        self.features1.add_module("relu_2", nn.ReLU())
        self.features1.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, dilation=1))
        self.features1.add_module("conv_3", nn.Conv2d(chan2, chan3, kernel_size=2))
        self.features1.add_module("relu_3", nn.ReLU())

        self.features2 = nn.Sequential()
        self.features2.add_module("conv_1", nn.Conv2d(1, chan1, kernel_size=3))
        self.features2.add_module("relu_1", nn.ReLU())
        self.features2.add_module("maxpool_1", nn.MaxPool2d(kernel_size=2))
        self.features2.add_module("conv_2", nn.Conv2d(chan1, chan2, kernel_size=2))
        self.features2.add_module("relu_2", nn.ReLU())
        self.features2.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, dilation=1))
        self.features2.add_module("conv_3", nn.Conv2d(chan2, chan3, kernel_size=2))
        self.features2.add_module("relu_3", nn.ReLU())

        class_size = self.features1(torch.empty((1, 1, 14, 14))).shape
        self.linear_size = reduce(multiplicator, list(class_size[1:]))

        self.classifier1 = nn.Sequential()
        self.classifier1.add_module("linear_1", nn.Linear(self.linear_size, nb_hidden1))
        self.classifier1.add_module("bn_1", nn.BatchNorm1d(nb_hidden1))
        self.classifier1.add_module("relu_1", nn.ReLU())
        self.classifier1.add_module("dropout_1", nn.Dropout(0.25))
        self.classifier1.add_module("linear_2", nn.Linear(nb_hidden1, nb_hidden2))
        self.classifier1.add_module("bn_2", nn.BatchNorm1d(nb_hidden2))
        self.classifier1.add_module("relu_2", nn.ReLU())
        self.classifier1.add_module("dropout_2", nn.Dropout(0.25))
        self.classifier1.add_module("linear_3", nn.Linear(nb_hidden2, 10))

        self.classifier2 = nn.Sequential()
        self.classifier2.add_module("linear_1", nn.Linear(self.linear_size, nb_hidden1))
        self.classifier2.add_module("bn_1", nn.BatchNorm1d(nb_hidden1))
        self.classifier2.add_module("relu_1", nn.ReLU())
        self.classifier2.add_module("dropout_1", nn.Dropout(0.25))
        self.classifier2.add_module("linear_2", nn.Linear(nb_hidden1, nb_hidden2))
        self.classifier2.add_module("bn_2", nn.BatchNorm1d(nb_hidden2))
        self.classifier2.add_module("relu_2", nn.ReLU())
        self.classifier2.add_module("dropout_2", nn.Dropout(0.25))
        self.classifier2.add_module("linear_3", nn.Linear(nb_hidden2, 10))

        self.classifierf = nn.Sequential()
        self.classifierf.add_module("linear_1", nn.Linear(20, nb_hidden3))
        self.classifierf.add_module("bn_1", nn.BatchNorm1d(nb_hidden3))
        self.classifierf.add_module("relu_1", nn.ReLU())
        self.classifierf.add_module("dropout_1", nn.Dropout(0.25))
        self.classifierf.add_module("linear_2", nn.Linear(nb_hidden3, nb_hidden4))
        self.classifierf.add_module("bn_2", nn.BatchNorm1d(nb_hidden4))
        self.classifierf.add_module("relu_2", nn.ReLU())
        self.classifierf.add_module("dropout_2", nn.Dropout(0.25))
        self.classifierf.add_module("linear_3", nn.Linear(nb_hidden4, 2))

    def forward(self, x):
        out_aux = []

        x_1 = x[:, 0, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        x_1 = self.features1(x_1)
        out_aux.append(self.classifier1(x_1.view(-1, self.linear_size)))

        x_2 = x[:, 1, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        x_2 = self.features2(x_2)
        out_aux.append(self.classifier2(x_2.view(-1, self.linear_size)))

        diff = torch.cat((out_aux[1], out_aux[0]), 1)
        out = self.classifierf(diff)
        return out, out_aux
