import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
from itertools import islice
import pandas as pd
# from load_data import load_three_data
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def my_plot(dir):
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
    plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('{} Parameters'.format('generalization err v.s. comlexity'), fontsize=16)
    plt.plot(x, population_risk_list - empirical_risk_list, 'darkturquoise', label='generalization err', linestyle='--', linewidth=2)
    plt.plot(x, complexity_list, 'blue', label='complexity_list', linewidth=2)
    plt.legend(fontsize=14, loc='best')

    plt.figure(3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('{} Parameters'.format('train acc v.s. val acc'), fontsize=16)
    plt.plot(x, train_acc_list, 'darkturquoise', label='train acc',linestyle='--', linewidth=2)
    plt.plot(x, val_acc_list, 'blue', label='val acc', linewidth=2)
    plt.legend(fontsize=14, loc='best')

if __name__ == '__main__':
    # dir = '../results/cifar10_non_iid_dirichlet_0.5_trainable_ConvNet4_10clients_seed:1_2023-08-30-23:14:25/'
    # dir = '../results/1000_epochs/medmnist_non_iid_dirichlet_0.2_trainable_ConvNet3_10clients_seed:2_2023-08-31-03:47:49/'
    # dir = '../results/1000_epochs/cifar10_non_iid_dirichlet_0.2_trainable_ConvNet4_10clients_seed:1_2023-08-31-01:58:22/'

    # dir = '../saved/medmnist_non_iid_dirichlet_0.1_trainable_ConvNet3_10clients_seed:1_2023-08-31-19:40:52/'
    # dir = '../saved/medmnist_custom_0.5_trainable_ConvNet3_10clients_seed:1_2023-08-31-20:23:20/'
    # dir = '../saved/cifar10_custom_0.1_trainable_ConvNet4_10clients_seed:1_2023-08-31-20:23:46/'
    dir = '../saved/cifar10_non_iid_dirichlet_0.5_trainable_ConvNet4_10clients_seed:1_2023-08-31-21:12:40/'
    my_plot(dir)
    plt.show()