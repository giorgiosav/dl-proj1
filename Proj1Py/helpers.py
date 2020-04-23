# -*- coding: utf-8 -*-
"""General helpers used in the implementation"""

import torch


def get_device():
    """
    Get the device in which all the tensor will be stored: CPU or CUDA (GPU) device
    :return: string defining the device
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def get_param_nums(model):
    """
    Get number of parameters in a model
    :param model: network model
    :type model: nn.Module
    :return: (int) number of parameters
    """
    # Compute and return params
    params = sum(p.numel() for p in model.parameters())
    return params


def get_best_params(model_name, optimizer="Adam", only_epochs=False):
    """
    Get pre-computed best hyper-parameters for a given model and optimizer
    :param model_name: name of model (Baseline/Siamese/NonSiamese)
    :type model_name: str
    :param optimizer: optimizer name (Adam/SGD)
    :type optimizer: str
    :param only_epochs:  only return number of epochs
    :type only_epochs: bool
    :return: dict of parameters, or int if only_epochs=True
    """

    # Get best parameters depending on model and optimizer
    if optimizer == "Adam":
        if model_name == "Baseline":
            if not only_epochs:
                return {"eta": 0.0025, "chan1": 32, "chan2": 64, "chan3": 128,
                        "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25,
                        "epochs": 25}
            else:
                return 25
        if model_name == "Siamese":
            if not only_epochs:
                return {"eta": 0.0025, "chan1": 64, "chan2": 128, "chan3": 256,
                        "nb_hidden1": 75, "nb_hidden2": 75, "nb_hidden3": 50, "nb_hidden4": 10,
                        "epochs": 25}
            else:
                return 25
        if model_name == "NonSiamese":
            if not only_epochs:
                return {"eta": 0.005, "chan1": 32, "chan2": 64, "chan3": 128,
                        "nb_hidden1": 75, "nb_hidden2": 75, "nb_hidden3": 50, "nb_hidden4": 10,
                        "epochs": 25}
            else:
                return 25
    else:
        if model_name == "Baseline":
            return {"eta": 0.1, "momentum": 0.01, "nesterov": False,
                    "chan1": 32, "chan2": 64, "chan3": 128,
                    "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25,
                    "epochs": 25}
        if model_name == "Siamese":
            return {"eta": 0.1, "momentum": 0.0001, "nesterov": False,
                    "chan1": 64, "chan2": 128, "chan3": 256,
                    "nb_hidden1": 75, "nb_hidden2": 75, "nb_hidden3": 50, "nb_hidden4": 10,
                    "epochs": 25}
        if model_name == "NonSiamese":
            return {"eta": 0.1, "momentum": 0.0001, "nesterov": False,
                    "chan1": 32, "chan2": 64, "chan3": 128,
                    "nb_hidden1": 75, "nb_hidden2": 75, "nb_hidden3": 50, "nb_hidden4": 10,
                    "epochs": 25}
