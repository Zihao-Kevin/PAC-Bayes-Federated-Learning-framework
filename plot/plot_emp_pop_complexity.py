import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
from itertools import islice
import pandas as pd
# from load_data import load_three_data
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def my_plot(dir, exp_name, dataset_name, partion_data_name):
    empirical_risk_list = np.load(dir + 'empirical_risk.npy', allow_pickle=True)[2:]
    population_risk_list = np.load(dir + 'population_risk.npy', allow_pickle=True)[2:]
    complexity_list = np.load(dir + 'complexity.npy', allow_pickle=True)[2:]
    train_acc_list = np.load(dir + 'train_acc.npy', allow_pickle=True)[2:]
    val_acc_list = np.load(dir + 'val_acc.npy', allow_pickle=True)[2:]

    x = range(1, empirical_risk_list.size + 1)

    plt.figure(1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('{} Parameters'.format('generalization err v.s. comlexity'), fontsize=16)
    plt.plot(x, population_risk_list - empirical_risk_list, 'darkturquoise', label='generalization err', linestyle='--', linewidth=2)

    plt.figure(2)
    complexity_max = np.max(population_risk_list - empirical_risk_list)
    # plt.ylim(0, complexity_max * 2)
    file_name = dataset_name + '_' + partion_data_name
    plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.xlabel("Global epoch", fontsize=14, labelpad=3)
    plt.yticks(fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=14, labelpad=0)
    plt.title('{} - generalization analysis'.format(exp_name), fontsize=16)
    plt.plot(x, complexity_list, 'blue', label='complexity', linewidth=2)
    plt.plot(x, population_risk_list - empirical_risk_list, 'darkturquoise', label='gen error', linewidth=2)
    plt.legend(fontsize=14, loc='best')
    plt.figure(2).savefig('{}_complexity.pdf'.format(file_name), dpi=1000, format='pdf')

    plt.figure(3)
    plt.xticks(fontsize=12)
    plt.xlabel("Global epoch", fontsize=14, labelpad=3)
    plt.yticks(fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=14, labelpad=0)
    plt.title('{} - model performance'.format(exp_name), fontsize=16)
    plt.plot(x, train_acc_list, 'darkturquoise', label='train acc', linewidth=2)
    plt.plot(x, val_acc_list, 'blue', label='val acc', linewidth=2)
    plt.legend(fontsize=14, loc='best')
    plt.figure(3).savefig('{}_train_val_acc.pdf'.format(file_name), dpi=1000, format='pdf')


if __name__ == '__main__':
    # dir = '../results/cifar10/noniid/cifar10_non_iid_dirichlet_0.1_trainable_ConvNet4_10clients_seed:1_2023-09-01-00:14:45/'
    dir = '../results/medmnist/noniid/medmnist_non_iid_dirichlet_0.1_trainable_ConvNet3_10clients_seed:1_2023-09-01-00:14:54/'
    # dataset_name = "CIFAR-10"
    dataset_name = "MedMNIST"
    partion_data_name = 'non-IID'
    exp_name = dataset_name + ' - ' + partion_data_name
    my_plot(dir, exp_name, dataset_name, partion_data_name)
    plt.show()