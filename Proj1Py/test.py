import torch
from data import get_data
from helpers import *
from models import *
from train import *
from validation import *


def main():
    train_loader, test_loader = get_data()
    device = get_device()
    model = BaseNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    epochs = 100
    eta = 0.002
    loss, acc = train_basic_model(model, train_loader, criterion, epochs, eta, get_losses=True, test_loader=test_loader)
    print(loss)
    print(acc)
    print(compute_accuracy_basic(model, train_loader))
    print(compute_accuracy_basic(model, test_loader))
    # chans = [32, 64, 128]
    # nb_hidden1 = [10, 25, 50, 75]
    # nb_hidden2 = nb_hidden3 = nb_hidden1
    # # nb_hidden1 = [50]
    # # nb_hidden2 = [50]
    # # nb_hidden3 = [25]
    # etas = [0.002]
    # # etas = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    # select_best_hyper_base(chans, nb_hidden1, nb_hidden2, nb_hidden3,
    #                        etas, momentum=[], nesterov=False, optim="Adam",
    #                        n_runs=5, epochs=15, verbose=True)



if __name__ == '__main__':
    main()
