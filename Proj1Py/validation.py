from helpers import get_device
from models import *
from train import *
from data import get_data


def select_best_hyper_base(chans, nb_hidden1, nb_hidden2, nb_hidden3,
                           etas, n_runs=10, epochs=25, verbose=False):
    device = get_device()
    best_acc = 0
    best_params = {"eta": 0, "chan1": 0, "chan2": 0, "chan3": 0, "nb_hidden1": 0, "nb_hidden2": 0, "nb_hidden3": 0}

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
                                    model = BaseNet(c1, c2, c3, h1, h2, h3).to(device)
                                    criterion = nn.CrossEntropyLoss().to(device)
                                    train_loader, test_loader = get_data()
                                    train_basic_model(model, train_loader, criterion, epochs, eta, optim="Adam")
                                    acc = compute_accuracy(model, test_loader)
                                    tot_acc += acc
                                    del model
                                acc_run = tot_acc / n_runs
                                if verbose:
                                    print("Eta = {}, chan1 = {}, chan2 = {}, chan3 = {}, "
                                          "hidden1 = {}, hidden2 = {}, hidden3 = {}, avg_acc = {}".
                                          format(eta, c1, c2, c3, h1, h2, h3, acc_run))
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


def select_best_hyper_advanced(chans, nb_hidden1, nb_hidden2, nb_hidden3, nb_hidden4,
                               etas, model_sel="Siamese", n_runs=10, epochs=25, verbose=False):
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
                                        if model_sel == "Siamese":
                                            model = SiameseNet(c1, c2, c3, h1, h2, h3, h4).to(device)
                                        elif model_sel == "NonSiamese":
                                            model = NonSiameseNet(c1, c2, c3, h1, h2, h3, h4).to(device)
                                        else:
                                            raise Exception("Siamese/NonSiamese selector exception")
                                        criterion = nn.CrossEntropyLoss().to(device)
                                        train_loader, test_loader = get_data()
                                        train_advanced_models(model, train_loader, criterion, epochs, eta, optim="Adam")
                                        acc = compute_accuracy(model, test_loader, "Advanced")
                                        tot_acc += acc
                                        del model
                                    acc_run = tot_acc / n_runs
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
