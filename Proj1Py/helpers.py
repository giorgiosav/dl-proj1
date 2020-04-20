import torch


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def get_param_nums(model):
    params = sum(p.numel() for p in model.parameters())
    return params


def get_best_params(model_name, optimizer="Adam", only_epochs=False):
    if optimizer == "Adam":
        if model_name == "Baseline":
            if not only_epochs:
                return {"eta": 0.002, "chan1": 32, "chan2": 64, "chan3": 128,
                        "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25,
                        "epochs": 25}
            else:
                return 25
        if model_name == "Siamese":
            if not only_epochs:
                return {"eta": 0.005, "chan1": 32, "chan2": 64, "chan3": 128,
                        "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25, "nb_hidden4": 50,
                        "epochs": 50}
            else:
                return 50
        if model_name == "NonSiamese":
            if not only_epochs:
                return {"eta": 0.005, "chan1": 32, "chan2": 64, "chan3": 128,
                        "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25, "nb_hidden4": 50,
                        "epochs": 50}
            else:
                return 50
    else:
        if model_name == "Baseline":
            return {"eta": 0.1, "momentum": 0.01, "nesterov": False,
                    "chan1": 32, "chan2": 64, "chan3": 128,
                    "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25,
                    "epochs": 25}
        if model_name == "Siamese":
            return {"eta": 0.005, "chan1": 32, "chan2": 64, "chan3": 128,
                    "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25, "nb_hidden4": 50,
                    "epochs": 50}
        if model_name == "NonSiamese":
            return {"eta": 0.005, "chan1": 32, "chan2": 64, "chan3": 128,
                    "nb_hidden1": 50, "nb_hidden2": 50, "nb_hidden3": 25, "nb_hidden4": 50,
                    "epochs": 50}
