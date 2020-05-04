# -*- coding: utf-8 -*-
"""Training functions and losses/accuracy computation on the models"""

import torch


def train_basic_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                      criterion: torch.nn.CrossEntropyLoss, epochs: int, eta: float,
                      optim: str = "Adam", momentum: float = 0,
                      nesterov: bool = False, get_losses: bool = False,
                      test_loader: torch.utils.data.DataLoader = None) -> tuple:
    """
    Train the baseline model with the chosen criterion, eta and epochs number.
    Eventually save and returns accuracy/losses per epoch.

    :param model: model to train
    :param train_loader: data loader to iterate over
    :param criterion: loss criterion
    :param epochs: number of epochs
    :param eta: learning rate
    :param optim: optimizer to use during training (Default Adam)
    :param momentum: momentum value for SGD optimizer
    :param nesterov: define if Nesterov algorithm should be used by SGD
    :param get_losses: define if intermediary accuracy and losses should be computed at each epoch for train/test
    :param test_loader: test data loader to iterate over
    :return losses, acc: losses and accuracy at each epochs, only if get_losses = True
    """

    # Create optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum, nesterov=nesterov)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    losses = {"train": [], "test": []}
    acc = {"train": [], "test": []}

    for e in range(0, epochs):
        # Start the traning
        loss_sum_train = 0
        for input_data, target_data, _ in iter(train_loader):
            # Do a step over a minibatch
            output = model(input_data)
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if get_losses:
                with torch.no_grad(): loss_sum_train += loss # Save partial loss

        # Get intermediate losses and accuracy
        if get_losses:
            losses['train'].append(loss_sum_train / (len(train_loader.dataset) / train_loader.batch_size))

            with torch.no_grad():
                acc['train'].append(compute_accuracy(model, train_loader))
                losses['test'].append(compute_losses_test_basic(model, test_loader, criterion))
                acc['test'].append(compute_accuracy(model, test_loader))

    if get_losses:
        return losses, acc


def compute_losses_test_basic(model: torch.nn.Module, test_loader: torch.utils.data.Dataloader,
                              criterion: torch.nn.CrossEntropyLoss) -> float:
    """
    Compute average total test loss per batch for the baseline
    :param model:network instance
    :param test_loader: data loader to iterate over (train or test)
    :param criterion: loss criterion
    :return: average total test loss metric
    """
    with torch.no_grad():
        loss_sum_test = 0
        for input_data, target_data, _ in iter(test_loader):
            output = model(input_data)
            loss = criterion(output, target_data)
            loss_sum_test += loss

    return loss_sum_test / (len(test_loader.dataset) / test_loader.batch_size)


def train_advanced_models(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                          criterion: torch.nn.CrossEntropyLoss, epochs: int, eta: float,
                          optim: str = "Adam", momentum: float = 0,
                          nesterov: bool = False, get_losses: bool = False,
                          test_loader: torch.utils.data.DataLoader = None) -> tuple:
    """
    Train the advanced networks (siamese or not siamese) with the chosen criterion, eta and epochs number
    Eventually save and returns accuracy/losses per epoch.

    :param model: model to train
    :param train_loader: data loader to iterate over
    :param criterion: loss criterion
    :param epochs: number of epochs
    :param eta: learning rate
    :param optim: optimizer to use during training (Default Adam)
    :param momentum: momentum value for SGD optimizer
    :param nesterov: define if Nesterov algorithm should be used by SGD
    :param get_losses: define if intermediary accuracy and losses should be computed at each epoch for train/test
    :param test_loader: test data loader to iterate over
    :return loss0, loss1, loss_class, loss_tot, acc: auxiliaty, total loss, and accuracy at each epochs,
    only if get_losses = True
    """

    # Define optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum, nesterov=nesterov)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    loss0 = {"train": [], "test": []}
    loss1 = {"train": [], "test": []}
    loss_class = {"train": [], "test": []}
    loss_tot = {"train": [], "test": []}
    acc = {"train": [], "test": []}

    for e in range(0, epochs):
        loss_sum0_train = 0
        loss_sum1_train = 0
        loss_sumclass_train = 0
        loss_sumtot_train = 0

        # Start training
        model.train(True)
        for input_data, target_data, class_data in iter(train_loader):
            output, out_aux = model(input_data)
            # Compute auxiliary losses
            loss_out = criterion(output, target_data)
            loss_aux0 = criterion(out_aux[0], class_data[:, 0])
            loss_aux1 = criterion(out_aux[1], class_data[:, 1])
            # Compute total loss and do the step
            loss = loss_out + loss_aux0 + loss_aux1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save train loss per batch if required
            if get_losses:
                with torch.no_grad():
                    loss_sum0_train += loss_aux0
                    loss_sum1_train += loss_aux1
                    loss_sumclass_train += loss_out
                    loss_sumtot_train += loss

        # Compute auxiliary losses, accuracy at this epoch for train and test
        if get_losses:
            loss0['train'].append(loss_sum0_train / (len(train_loader.dataset) / train_loader.batch_size))
            loss1['train'].append(loss_sum1_train / (len(train_loader.dataset) / train_loader.batch_size))
            loss_class['train'].append(loss_sumclass_train / (len(train_loader.dataset) / train_loader.batch_size))
            loss_tot['train'].append(loss_sumtot_train / (len(train_loader.dataset) / train_loader.batch_size))

            with torch.no_grad():
                model.train(False)
                acc['train'].append(compute_accuracy(model, train_loader, "Advanced"))
                loss_0_test, loss_1_test, loss_class_test, loss_tot_test = \
                    compute_losses_test_advanced(model, test_loader, criterion)
                loss0['test'].append(loss_0_test)
                loss1['test'].append(loss_1_test)
                loss_class['test'].append(loss_class_test)
                loss_tot['test'].append(loss_tot_test)
                acc['test'].append(compute_accuracy(model, test_loader, "Advanced"))

    if get_losses:
        return loss0, loss1, loss_class, loss_tot, acc


def compute_losses_test_advanced(model: torch.nn.Module, test_loader: torch.utils.data.Dataloader,
                                 criterion: torch.nn.CrossEntropyLoss):
    """
    Compute average total test loss per batch for the advanced network
    :param model:network instance
    :param test_loader: data loader to iterate over (train or test)
    :param criterion: loss criterion
    :return: average total test loss metric
    """
    with torch.no_grad():
        loss_sum0_test = 0
        loss_sum1_test = 0
        loss_sumclass_test = 0
        loss_sumtot_test = 0
        for input_data, target_data, class_data in iter(test_loader):
            output, out_aux = model(input_data)
            loss_out = criterion(output, target_data)
            loss_aux0 = criterion(out_aux[0], class_data[:, 0])
            loss_aux1 = criterion(out_aux[1], class_data[:, 1])
            loss = loss_out + loss_aux0 + loss_aux1

            loss_sum0_test += loss_aux0
            loss_sum1_test += loss_aux1
            loss_sumclass_test += loss_out
            loss_sumtot_test += loss

    return loss_sum0_test / (len(test_loader.dataset) / test_loader.batch_size), \
           loss_sum1_test / (len(test_loader.dataset) / test_loader.batch_size), \
           loss_sumclass_test / (len(test_loader.dataset) / test_loader.batch_size), \
           loss_sumtot_test / (len(test_loader.dataset) / test_loader.batch_size)


def compute_accuracy(model: torch.nn.Module, data_loader: torch.utils.data.Dataloader,
                     model_type: str = "Baseline") -> float:
    """
    Compute number of errors for a network
    :param model:network instance
    :param data_loader: data loader to iterate over (train or test)
    :param model_type: define if accuracy is computed on baseline or Siamese/Not Siamese
    :return: accuracy metric
    """
    tot_err = 0
    for input_data, target_data, _ in iter(data_loader):
        # Get output from the model
        if model_type == "Baseline":
            res = model(input_data)
        else:
            model.train(False)
            res, _ = model(input_data)
        # Count the number of errors
        for i, r in enumerate(res):
            pred = r.max(0)[1].item()
            if (target_data[i]) != pred:
                tot_err += 1
    # Return accuracy
    return 1 - tot_err / (len(data_loader.dataset))
