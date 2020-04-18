import torch
from data import get_data
from helpers import *
from models import *
from train import *


def main():
    train_loader, test_loader = get_data()
    device = get_device()
    model = BaseNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    epochs = 75
    eta = 0.01
    loss, acc = train_basic_model(model, train_loader, criterion, epochs, eta, get_losses=True, test_loader=test_loader)
    print(loss)
    print(acc)
    print(compute_accuracy_basic(model, train_loader))
    print(compute_accuracy_basic(model, test_loader))


if __name__ == '__main__':
    main()
