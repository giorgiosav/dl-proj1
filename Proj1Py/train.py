import torch


def train_basic_model(model, train_loader, criterion, epochs, eta,
                      optim="Adam", momentum=0, nesterov=False, get_losses=False, test_loader=None):
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum, nesterov=nesterov)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    losses = {"train": [], "test": []}
    acc = {"train": [], "test": []}

    for e in range(0, epochs):
        loss_sum_train = 0
        for input_data, target_data, _ in iter(train_loader):
            output = model(input_data)
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if get_losses:
                with torch.no_grad(): loss_sum_train += loss

        if get_losses:
            losses['train'].append(loss_sum_train / (len(train_loader.dataset) / train_loader.batch_size))

            with torch.no_grad():
                acc['train'].append(compute_accuracy(model, train_loader))
                losses['test'].append(compute_losses_test_basic(model, test_loader, criterion))
                acc['test'].append(compute_accuracy(model, test_loader))

    if get_losses:
        return losses, acc


def compute_accuracy_basic(model, data_loader):
    """
    :param model:
    :param data_loader:
    :type data_loader: torch.utils.data.dataloader.DataLoader
    :return:
    """
    tot_err = 0
    for input_data, target_data, _ in iter(data_loader):
        res = model(input_data)
        for i, r in enumerate(res):
            pred = r.max(0)[1].item()
            if (target_data[i]) != pred:
                tot_err += 1
    return 1 - tot_err / (len(data_loader.dataset))


def compute_losses_test_basic(model, test_loader, criterion):
    with torch.no_grad():
        loss_sum_test = 0
        for input_data, target_data, _ in iter(test_loader):
            output = model(input_data)
            loss = criterion(output, target_data)
            loss_sum_test += loss

    return loss_sum_test / (len(test_loader.dataset) / test_loader.batch_size)


def train_advanced_models(model, train_loader, criterion, epochs, eta,
                          optim="Adam", momentum=0, nesterov=False, get_losses=False, test_loader=None):
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

        model.train(True)
        for input_data, target_data, class_data in iter(train_loader):
            output, out_aux = model(input_data)
            loss_out = criterion(output, target_data)
            loss_aux0 = criterion(out_aux[0], class_data[:, 0])
            loss_aux1 = criterion(out_aux[1], class_data[:, 1])
            loss = loss_out + loss_aux0 + loss_aux1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if get_losses:
                with torch.no_grad():
                    loss_sum0_train += loss_aux0
                    loss_sum1_train += loss_aux1
                    loss_sumclass_train += loss_out
                    loss_sumtot_train += loss

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


def compute_losses_test_advanced(model, test_loader, criterion):
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


def compute_accuracy(model, data_loader, model_type="Baseline"):
    """
    :param model_type:
    :param model:
    :param data_loader:
    :type data_loader: torch.utils.data.dataloader.DataLoader
    :return:
    """
    tot_err = 0
    for input_data, target_data, _ in iter(data_loader):
        if model_type == "Baseline":
            res = model(input_data)
        else:
            model.train(False)
            res, _ = model(input_data)
        for i, r in enumerate(res):
            pred = r.max(0)[1].item()
            if (target_data[i]) != pred:
                tot_err += 1
    return 1 - tot_err / (len(data_loader.dataset))
