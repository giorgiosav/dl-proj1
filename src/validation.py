# -*- coding: utf-8 -*-
"""Validation algorithms to compute best hyperparameters"""

from helpers import get_device
from models import *
from train import *
from data import get_data


def select_best_hyper_base(chans: list, nb_hidden1: list, nb_hidden2: list, nb_hidden3: list,
                           etas: list, n_runs: int = 10, epochs: int = 25, verbose: bool = False) -> dict:
    """
    Partial grid search over hyper-parameters for the baseline network, to find
    the best combination. This function makes sure that only combinations with
    number of channels that increase with depth are tested.

    :param chans:  number of channels for each convolution layer (3)
    :param nb_hidden1: number of hidden units for 1st fully connected layer
    :param nb_hidden2: number of hidden units for 2nd fully connected layer
    :param nb_hidden3: number of hidden units for 3rd fully connected layer
    :param etas: eta values to test
    :param n_runs: number of runs over which to average results
    :param epochs: number of epochs
    :param verbose: activate verbose printing
    :return: best_params, dict of best parameters found
    """
    device = get_device()
    best_acc = 0
    best_params = {"eta": 0, "chan1": 0, "chan2": 0, "chan3": 0, "nb_hidden1": 0, "nb_hidden2": 0, "nb_hidden3": 0}

    # Start grid search
    for eta in etas:
        for i1 in range(len(chans) - 2):
            c1 = chans[i1]
            for i2 in range(i1 + 1, len(chans) - 1):
                c2 = chans[i2]
                for i3 in range(i2 + 1, len(chans)):
                    c3 = chans[i3]
                    for h1 in nb_hidden1:
                        for h2 in nb_hidden2:
                            for h3 in nb_hidden3:
                                tot_acc = 0
                                for i in range(0, n_runs):
                                    # Create net, train it and compute accuracy on test data
                                    model = BaseNet(c1, c2, c3, h1, h2, h3).to(device)
                                    criterion = nn.CrossEntropyLoss().to(device)
                                    # A new train/test set is used at each run to avoid overfitting a dataset
                                    train_loader, test_loader = get_data()
                                    train_basic_model(model, train_loader, criterion, epochs, eta, optim="Adam")
                                    acc = compute_accuracy(model, test_loader)
                                    tot_acc += acc
                                    del model
                                # Compute accuracy of the params set
                                acc_run = tot_acc / n_runs
                                if verbose:
                                    print("Eta = {}, chan1 = {}, chan2 = {}, chan3 = {}, "
                                          "hidden1 = {}, hidden2 = {}, hidden3 = {}, avg_acc = {}".
                                          format(eta, c1, c2, c3, h1, h2, h3, acc_run))
                                # Save params
                                if acc_run > best_acc:
                                    best_acc = acc_run
                                    best_params["eta"] = eta
                                    best_params["chan1"] = c1
                                    best_params["chan2"] = c2
                                    best_params["chan3"] = c3
                                    best_params["nb_hidden1"] = h1
                                    best_params["nb_hidden2"] = h2
                                    best_params["nb_hidden3"] = h3
                                    if verbose:
                                        print("New best combination: Eta = {}, chan1 = {}, chan2 = {}, chan3 = {}, "
                                              "hidden1 = {}, hidden2 = {}, hidden3 = {}, avg_acc = {}"
                                              .format(eta, c1, c2, c3, h1, h2, h3, acc_run))

    print("Best result found! Acc: {}, "
          "params: Eta = {}, chan1 = {}, "
          "chan2 = {}, chan3 = {}, "
          "hidden1 = {}, hidden2 = {}, hidden3 = {}"
          .format(best_acc, best_params["eta"], best_params["chan1"], best_params["chan2"],
                  best_params["chan3"], best_params["nb_hidden1"],
                  best_params["nb_hidden2"], best_params["nb_hidden3"]))
    return best_params


def select_best_hyper_advanced(chans: list, nb_hidden1: list, nb_hidden2: list, nb_hidden3: list, nb_hidden4: list,
                               etas: list, model_sel: str = "Siamese",
                               n_runs: int = 10, epochs: int = 25, verbose: bool = False) -> dict:
    """
    Partial grid search over hyper-parameters for the siamese and non-siamese networks, to find
    the best combination. This function makes sure that only combinations with
    number of channels that increase with depth are tested.

    :param chans:  number of channels for each convolution layer (3)
    :param nb_hidden1: number of hidden units for 1st fully connected layer
    :param nb_hidden2: number of hidden units for 2nd fully connected layer
    :param nb_hidden3: number of hidden units for 3rd fully connected layer
    :param nb_hidden4: number of hidden units for 3rd fully connected layer
    :param etas: eta values to test
    :param model_sel: type of model between siamese and not siamese
    :param n_runs: number of runs over which to average results
    :param epochs: number of epochs
    :param verbose: activate verbose printing
    :return: best_params, dict of best parameters found
    """

    device = get_device()
    best_acc = 0
    best_params = {"eta": 0, "chan1": 0, "chan2": 0, "chan3": 0,
                   "nb_hidden1": 0, "nb_hidden2": 0, "nb_hidden3": 0, "nb_hidden4": 0}

    for eta in etas:
        for i1 in range(len(chans) - 2):
            c1 = chans[i1]
            for i2 in range(i1 + 1, len(chans) - 1):
                c2 = chans[i2]
                for i3 in range(i2 + 1, len(chans)):
                    c3 = chans[i3]
                    for h1 in nb_hidden1:
                        for h2 in nb_hidden2:
                            for h3 in nb_hidden3:
                                for h4 in nb_hidden4:
                                    tot_acc = 0
                                    for i in range(0, n_runs):
                                        # Create net, train it and compute accuracy on test data
                                        if model_sel == "Siamese":
                                            model = SiameseNet(c1, c2, c3, h1, h2, h3, h4).to(device)
                                        elif model_sel == "NonSiamese":
                                            model = NonSiameseNet(c1, c2, c3, h1, h2, h3, h4).to(device)
                                        else:
                                            raise Exception("Siamese/NonSiamese selector exception")
                                        criterion = nn.CrossEntropyLoss().to(device)
                                        # A new train/test set is used at each run to avoid overfitting a dataset
                                        train_loader, test_loader = get_data()
                                        train_advanced_models(model, train_loader, criterion, epochs, eta, optim="Adam")
                                        acc = compute_accuracy(model, test_loader, "Advanced")
                                        tot_acc += acc
                                        del model
                                    acc_run = tot_acc / n_runs
                                    # Save accuracy if better than current best
                                    if verbose:
                                        print("Eta = {}, chan1 = {}, chan2 = {}, chan3 = {}, "
                                              "hidden1 = {}, hidden2 = {}, hidden3 = {}, hidden4 = {}, avg_acc = {}".
                                              format(eta, c1, c2, c3, h1, h2, h3, h4, acc_run))
                                    if acc_run > best_acc:
                                        best_acc = acc_run
                                        best_params["eta"] = eta
                                        best_params["chan1"] = c1
                                        best_params["chan2"] = c2
                                        best_params["chan3"] = c3
                                        best_params["nb_hidden1"] = h1
                                        best_params["nb_hidden2"] = h2
                                        best_params["nb_hidden3"] = h3
                                        best_params["nb_hidden4"] = h4
                                        if verbose:
                                            print("New best combination: Eta = {}, chan1 = {}, chan2 = {}, chan3 = {}, "
                                                  "hidden1 = {}, hidden2 = {}, hidden3 = {}, hidden4 = {}, avg_acc = {}"
                                                  .format(eta, c1, c2, c3, h1, h2, h3, h4, acc_run))

    print("Best result found! Acc: {}, "
          "params: Eta = {}, chan1 = {}, "
          "chan2 = {}, chan3 = {}, "
          "hidden1 = {}, hidden2 = {}, hidden3 = {}, hidden4 = {}"
          .format(best_acc, best_params["eta"], best_params["chan1"], best_params["chan2"],
                  best_params["chan3"], best_params["nb_hidden1"],
                  best_params["nb_hidden2"], best_params["nb_hidden3"], best_params["nb_hidden4"]))
    return best_params
