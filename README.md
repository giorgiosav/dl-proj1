# EE-559 Deep Learning - Project 1

## Introduction
In this project we explored the problem of classifying pairs of MNIST images based on whether one number was larger than the other using deep neural network. 
In order to reach this goal, the baseline was set to a naive network with no training on the image classes. 
This network was then gradually improved by using mechanisms such as auxiliary losses and weight sharing, that led us to build a siamese network. 
The performance achieved by this network was satisfactory for the objective, with an overall accuracy of 94.5% over 25 epochs of training performed in 7.59 s.

## Prerequites
### Basic prerequisites
The minimum requirements for the `test.py` are:
- `Python` (tested on version **_3.8_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version *20.2.3*) (For package installation, if needed)
- `Pytorch` (tested on version *1.6.0*)
- `Matplotlib` (tested on version *3.3.2*)

**NOTE**: If you have LaTeX installed on your machine, you can uncomment lines `plot.py:11-17` to produce plots with a "LaTeX style". 

## Usage instruction
1. Open CMD/Bash
2. Activate the environment with needed packages installed (if you have Anaconda or any virtual environment on your machine)
3. Move to the src folder, where the `test.py` is located
4. Execute the command ```python test.py``` with zero or more of the following arguments:
```
Optional arguments:
  -validation     Run validation on the chosen model. If not set, already
                  selected best parameters will be used. (default false)
  -sgd            Train the selected model with SGD instead of Adam. The best
                  parameters will be automatically loaded. Note: the performance
                  of SGD are tested as worse and slower,for this reason the
                  validation algorithm is available only for Adam optimizer (default false)
  -plots          Create the accuracy/loss plot over epochs for the selected
                  model as shown in the report. This option deactivates
                  printing of mean training time, as it creates overhead. (deafult false)
  -n_runs N_RUNS  Define number of runs of the train/test process with the
                  selected model (default 10)

  Possible models:
    Select one of the three possible models

    -baseline       Run Baseline model
    -siamese        Run Siamese model (default model)
    -notsiam        Run Non Siamese model
 
```

## Results
Detailed explanation of the results we achieved can be found in our [Report](https://github.com/giorgiosav/dl-proj1/blob/master/Project1_Report.pdf) and inside the
[plot folder](https://github.com/giorgiosav/dl-proj1/tree/master/src/plot), where you can find the accuracy and the loss we achieved with our implementations (Baseline, Siamese and Not Siamese networks).

## Folder structure
```
.
├── extra_stuff
│   ├── ee559-miniprojects.pdf
│   └── Notebooks.zip
├── Project1_Report.pdf
├── README.md
└── src
    ├── data.py
    ├── dlc_practical_prologue.py
    ├── helpers.py
    ├── models.py
    ├── plot
    │   ├── acc_Baseline_15runs.pdf
    │   ├── acc_NonSiamese_15runs.pdf
    │   ├── acc_Siamese_15runs.pdf
    │   ├── losstot_Baseline_15runs.pdf
    │   ├── losstot_NonSiamese_15runs.pdf
    │   └── losstot_Siamese_15runs.pdf
    ├── plot.py
    ├── test.py
    ├── train.py
    └── validation.py

```

## Code reproducibility
Even if all the random seeds are set in the `test.py`, `Pytorch` always has a bit of randomness.
For this reason, the reader is advised that **different runs with the same parameters, and also different runs of CV can produce slightly different results**.  

## Authors
- [Manuel Leone](https://github.com/manuleo)
- [Giorgio Savini](https://github.com/giorgiosav)
