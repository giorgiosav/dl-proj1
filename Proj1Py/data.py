# -*- coding: utf-8 -*-
"""Get the data from the prologue and convert them to dataloader for easier use"""

from dlc_practical_prologue import generate_pair_sets
import torch
from torch.utils.data import DataLoader, Dataset
from helpers import *


class MNISTCompareDataset(Dataset):
    """
    Convert the tensor obtained from the prologue into a Dataset, to use them into a Dataloader
    """
    def __init__(self, input_data: torch.Tensor, target_data: torch.Tensor, classes_data: torch.Tensor):
        """
        :param input_data = dataset of MNIST images
        :param target_data = dataset of MNIST images
        :param classes_data = dataset of MNIST images
        """
        self.input_data = input_data
        self.target_data = target_data
        self.classes_data = classes_data

    def __len__(self) -> int:
        """
        Get length
        :return: length of the dataset
        """
        return len(self.input_data)

    def __getitem__(self, i: int) -> tuple:
        """
        Get data at position i
        :param i: index to get data, target and classes
        :return: data sample, target class, classes of each input
        """
        data = self.input_data[i, :, :, :]
        target = self.target_data[i]
        classes = self.classes_data[i, :]
        return data, target, classes


def get_data(N: int = 1000, batch_size: int = 100, shuffle: bool = True) -> tuple:
    """
    Get train and test DataLoaders of size N
    :param N: number of pairs to return for each data loader
    :param batch_size: batch size
    :param shuffle: activate random shuffling
    :return: train and test loader
    """

    # Get input tensors from provided prologue
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N)
    device = get_device()

    # Normalization
    mu, std = train_input.mean(), train_input.std()
    train_input = train_input.sub(mu).div(std)
    test_input = test_input.sub(mu).div(std)

    # Move data to GPU if available
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_classes = train_classes.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    test_classes = test_classes.to(device)

    # Create train and test loader
    train_loader = DataLoader(MNISTCompareDataset(train_input, train_target, train_classes),
                              batch_size=batch_size, shuffle=shuffle)

    test_loader = DataLoader(MNISTCompareDataset(test_input, test_target, test_classes),
                             batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
