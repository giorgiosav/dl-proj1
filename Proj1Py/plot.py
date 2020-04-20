import matplotlib.pyplot as plt
import matplotlib
import torch

# Used to save in Latex design
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot_over_epochs(values_list, epochs, label, savename):
    mean_train = torch.mean(torch.Tensor([val['train'] for val in values_list]), 0)
    mean_test = torch.mean(torch.Tensor([val['test'] for val in values_list]), 0)
    err_up_train = torch.max(torch.Tensor([val['train'] for val in values_list]), 0)[0] - mean_train
    err_down_train = mean_train - torch.min(torch.Tensor([val['train'] for val in values_list]), 0)[0]
    err_up_test = torch.max(torch.Tensor([val['test'] for val in values_list]), 0)[0] - mean_test
    err_down_test = mean_test - torch.min(torch.Tensor([val['test'] for val in values_list]), 0)[0]
    epochs_range = range(0, epochs)

    plt.figure()

    err_train = [err_down_train, err_up_train]
    err_test = [err_down_test, err_up_test]
    markers, caps, bars = plt.errorbar(epochs_range, mean_train, yerr=err_train, label="Train " + label,
                                       color="blue")
    [bar.set_alpha(0.5) for bar in bars]
    markers, caps, bars = plt.errorbar(epochs_range, mean_test, yerr=err_test, label="Test " + label,
                                       color="orange")
    [bar.set_alpha(0.5) for bar in bars]
    plt.xticks(range(0, epochs, 2))

    # set labels (LaTeX can be used)
    plt.xlabel(r'\textbf{Epochs}', fontsize=11)
    plt.ylabel(r'\textbf{' + label + '}', fontsize=11)
    plt.legend()
    plt.savefig("plot/" + savename + ".pdf")
    plt.close()
