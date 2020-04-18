import torch
from operator import mul as multiplicator
from functools import reduce


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def get_param_nums(model):
    params = sum(p.numel() for p in model.parameters())
    return params
