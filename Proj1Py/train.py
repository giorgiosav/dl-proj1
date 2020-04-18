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
            losses['train'].append(loss_sum_train / len(train_loader))

            with torch.no_grad():
                acc['train'].append(compute_accuracy_basic(model, train_loader))
                losses['test'].append(compute_losses_test_basic(model, test_loader, criterion))
                acc['test'].append(compute_accuracy_basic(model, test_loader))

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
    return 1 - tot_err / (len(data_loader) * data_loader.batch_size)


def compute_losses_test_basic(model, test_loader, criterion):
    with torch.no_grad():
        loss_sum_test = 0
        for input_data, target_data, _ in iter(test_loader):
            output = model(input_data)
            loss = criterion(output, target_data)
            loss_sum_test += loss

    return loss_sum_test / len(test_loader)
