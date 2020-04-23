# -*- coding: utf-8 -*-
"""Main entry point to test the networks implementation"""

from helpers import *
from validation import *
from plot import *
import argparse
import sys
import time


def test_selected_model(model_name, sgd, plots, best_params, n_runs):
    """
    Test the selected model by running train/test over the defined number of runs and using selected optimizer.
    Eventually plot loss/accuracy over epochs.

    :param model_name: name of network (Baseline/Siamese/NonSiamese)
    :type model_name: str
    :param sgd: use SGD if True, else Adam
    :type sgd: bool
    :param plots: generate loss/accuracy plots
    :type plots: bool
    :param best_params: hyper parameters to use
    :type best_params: dict
    :param n_runs: number of times to run
    :type n_runs: int
    """
    print("Starting the train/test phase over {} runs".format(n_runs))
    if plots:
        print("\nYou chose to produce the plots. \n"
              "Model will be trained saving losses/accuracy at each epochs, this will require more time.\n"
              "For this reason, the average training time won't be computed (otherwise the estimate will be biased)\n")

    # Get params and define lists to save intermediate results
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    epochs = best_params["epochs"]
    eta = best_params["eta"]
    loss_tot = []
    acc_tot = []
    acc_train = []
    acc_test = []
    train_time_acc = 0
    for i in range(0, n_runs):
        print("Run {}".format(i))
        # Get random data and create net
        train_loader, test_loader = get_data()
        if model_name == "Baseline":
            model = BaseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                            nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                            nb_hidden3=best_params["nb_hidden3"])
        elif model_name == "Siamese":
            model = SiameseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                               nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                               nb_hidden3=best_params["nb_hidden3"], nb_hidden4=best_params["nb_hidden4"])
        else:
            model = NonSiameseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                                  nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                                  nb_hidden3=best_params["nb_hidden3"], nb_hidden4=best_params["nb_hidden4"])
        model = model.to(device)
        # Define optimizer
        if sgd:
            optim = "SGD"
            nesterov = best_params["nesterov"]
            momentum = best_params["momentum"]
        else:
            optim = "Adam"
            nesterov = None
            momentum = None
        print("Training model {}...".format(i))
        if plots:
            # Train saving data later used for plots
            if model_name == "Baseline":
                loss, acc = train_basic_model(model, train_loader, criterion, epochs, eta,
                                              optim=optim, momentum=momentum, nesterov=nesterov,
                                              get_losses=True, test_loader=test_loader)
            else:
                # Auxiliary losses won't be plotted, they were computed
                # only for debug/improvement purpose during developing
                _, _, _, loss, acc = train_advanced_models(model, train_loader, criterion, epochs, eta,
                                                           optim=optim, momentum=momentum, nesterov=nesterov,
                                                           get_losses=True, test_loader=test_loader)
            loss_tot.append(loss)
            acc_tot.append(acc)
        else:
            # Do training without saving data, compute training time
            start_time = time.time()
            if model_name == "Baseline":
                train_basic_model(model, train_loader, criterion, epochs, eta,
                                  optim=optim, momentum=momentum, nesterov=nesterov)
            else:
                train_advanced_models(model, train_loader, criterion, epochs, eta,
                                      optim=optim, momentum=momentum, nesterov=nesterov)
            end_time = time.time()
            train_time_acc += end_time - start_time

        print("Training on model {} finished, computing accuracy on train and test...".format(i))
        # Compute accuracy on train and test
        acc_train.append(compute_accuracy(model, train_loader, model_name))
        acc_test.append(compute_accuracy(model, test_loader, model_name))
        del model

    if plots:
        # Creating plots and saving them to pdf files
        print("-------------------------------------------------------")
        print("Saving requested plots for loss and accuracy")
        loss_save = "losstot_{model}_{n}runs{sgd}".format(model=model_name, n=n_runs, sgd="_sgd" if sgd else "")
        acc_save = "acc_{model}_{n}runs{sgd}".format(model=model_name, n=n_runs, sgd="_sgd" if sgd else "")
        plot_over_epochs(loss_tot, epochs, "Loss", loss_save)
        plot_over_epochs(acc_tot, epochs, "Accuracy", acc_save)

    # Computing mean accuracy, std and mean train time over the runs
    mean_acc_train = torch.mean(torch.Tensor(acc_train))
    mean_acc_test = torch.mean(torch.Tensor(acc_test))
    var_acc_train = torch.std(torch.Tensor(acc_train))
    var_acc_test = torch.std(torch.Tensor(acc_test))
    print("-------------------------------------------------------")
    mean_train_time = train_time_acc / n_runs
    print("------------------------------------------")
    print("Final accuracy and standard deviation on train and test:")
    print("Train -> Mean Accuracy = {}, Standard deviation = {}".format(mean_acc_train, var_acc_train))
    if not plots:
        print("      -> Mean Train Time = {:.3}s".format(mean_train_time))
    print("Test -> Mean Accuracy = {}, Standard deviation = {}".format(mean_acc_test, var_acc_test))

    return


def main(validation, sgd, model_name, plots, n_runs):
    """
    :param validation: perform grid search for best parameters
    :type validation: bool
    :param model_name: name of network (Baseline/Siamese/NonSiamese)
    :type model_name: str
    :param sgd: use SGD if True, else Adam
    :type sgd: bool
    :param plots: generate loss/accuracy plots
    :type plots: bool
    :param n_runs: number of times to run
    :type n_runs: int
    """

    # If no arguments are received, print help info
    if len(sys.argv) == 1:
        print("\n-------------------------------------------------------")
        print("No arguments defined. Default best configuration used.\n"
              "To receive help on how to set parameters \nand run different implementations\n"
              "use the command: python test.py -h")
        print("-------------------------------------------------------")

    print("{} model implementation".format(model_name))

    # Perform validation
    if validation:
        # The grid search is performed over parameters
        # we already found performing better during coarse grained validation
        if not sgd:
            chans = [32, 64, 128, 256]
            nb_hidden1 = nb_hidden2 = nb_hidden3 = nb_hidden4 = [10, 25, 50, 75, 100]
            etas = [0.001, 0.0025, 0.005, 0.0075, 0.01]
            print("Starting validation algorithm on the chosen model. "
                  "NOTE: this may require up to 10 hours for a complete run")
            if model_name == "Baseline":
                best_params = select_best_hyper_base(chans, nb_hidden1, nb_hidden2, nb_hidden3, etas)
            else:
                best_params = select_best_hyper_advanced(chans, nb_hidden1, nb_hidden2, nb_hidden3, nb_hidden4,
                                                         etas, model_name, 5, 30, True)

            # The best number of epochs was determined by plot analysis, so we add it manually
            best_params["epochs"] = get_best_params(model_name, only_epochs=True)
        else:
            print("Validation not available for SGD optimizer, loading precomputed best params")
            best_params = get_best_params(model_name, "SGD")

    else:
        print("Loading precomputed best params")
        if sgd:
            best_params = get_best_params(model_name, "SGD")
        else:
            best_params = get_best_params(model_name)
    print("-------------------------------------------------------")

    # Print number of parameters in the chosen model
    if model_name == "Baseline":
        model = BaseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                            nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                            nb_hidden3=best_params["nb_hidden3"])
    elif model_name == "Siamese":
        model = SiameseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                            nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                            nb_hidden3=best_params["nb_hidden3"], nb_hidden4=best_params["nb_hidden4"])
    else:
        model = NonSiameseNet(chan1=best_params["chan1"], chan2=best_params["chan2"], chan3=best_params["chan3"],
                            nb_hidden1=best_params["nb_hidden1"], nb_hidden2=best_params["nb_hidden2"],
                            nb_hidden3=best_params["nb_hidden3"], nb_hidden4=best_params["nb_hidden4"])
    params_num = get_param_nums(model)
    del model
    print("Number of parameters of the model: {}".format(params_num))
    print("-------------------------------------------------------")

    # Run train/test over selected number of rounds
    test_selected_model(model_name, sgd, plots, best_params, n_runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the DL Project 1 implementation.')
    # Loading model params
    parser.add_argument('-validation', action='store_true',
                        help="Run validation on the chosen model. "
                             "If not set, already selected best parameters will be used.")

    parser.add_argument('-sgd', action='store_true', help="Train the selected model with SGD instead of Adam. "
                                                          "The best parameters will be automatically loaded"
                                                          "Note: the performance of SGD are tested as worse and slower,"
                                                          "for this reason the validation algorithm is available only "
                                                          "for Adam optimizer")

    title_models = parser.add_argument_group('Possible models', 'Select one of the three possible models')
    group_models = title_models.add_mutually_exclusive_group()
    group_models.add_argument('-baseline', action='store_const', help="Run Baseline model", dest="model",
                              const="Baseline")
    group_models.add_argument('-siamese', action='store_const', help="Run Siamese model (default model)", dest="model",
                              const="Siamese")
    group_models.add_argument('-notsiam', action='store_const', help="Run Non Siamese model", dest="model",
                              const="NonSiamese")

    parser.add_argument('-plots', action='store_true',
                        help="Create the accuracy/loss plot over epochs for the selected model as shown in the report. "
                             "This option deactivates printing of mean training time, as it creates overhead.")

    parser.add_argument('-n_runs', help='Define number of runs of the train/test process '
                                        'with the selected model (default 10)',
                        type=int, default=10)

    parser.set_defaults(model='Siamese')
    args = parser.parse_args()

    main(args.validation, args.sgd, args.model, args.plots, args.n_runs)
