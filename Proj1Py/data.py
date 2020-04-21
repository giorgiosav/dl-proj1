from dlc_practical_prologue import generate_pair_sets
import torch
from torch.utils.data import DataLoader, Dataset
from helpers import *


class MNISTCompareDataset(Dataset):
    def __init__(self, input_data, target_data, classes_data):
        """
        :param input_data = dataset of MNIST images
        :type input_data: torch.Tensor
        :param target_data = dataset of MNIST images
        :type target_data: torch.Tensor
        :param classes_data = dataset of MNIST images
        :type classes_data: torch.Tensor
        """
        self.input_data = input_data
        self.target_data = target_data
        self.classes_data = classes_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, i):
        data = self.input_data[i, :, :, :]
        target = self.target_data[i]
        classes = self.classes_data[i, :]
        return data, target, classes


def get_data(N=1000, batch_size=100, shuffle=True):
    '''
    Get train and test DataLoaders
    :param N: (int) number of pairs to return for each data loader
    :param batch_size: (int) batch size
    :param shuffle: (bool) activate random shuffling
    '''
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
    device = get_device()

    # Normalization
    mu, std = train_input.mean(), train_input.std()
    train_input = train_input.sub(mu).div(std)
    test_input = test_input.sub(mu).div(std)

    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_classes = train_classes.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    test_classes = test_classes.to(device)

    train_loader = DataLoader(MNISTCompareDataset(train_input, train_target, train_classes),
                              batch_size=batch_size, shuffle=shuffle)

    test_loader = DataLoader(MNISTCompareDataset(test_input, test_target, test_classes),
                             batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
