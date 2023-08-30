import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
from itertools import islice
import pandas as pd
# from load_data import load_three_data
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def my_plot(dir_fixed, dir_trainable, i_figure, tag):
    empirical_risk_list_fixed = np.load(dir_fixed + 'empirical_risk.npy', allow_pickle=True)
    population_risk_list_fixed = np.load(dir_fixed + 'population_risk.npy', allow_pickle=True)
    complexity_list_fixed = np.load(dir_fixed + 'complexity.npy', allow_pickle=True)
    train_acc_list_fixed = np.load(dir_fixed + 'train_acc.npy', allow_pickle=True)
    val_acc_list_fixed = np.load(dir_fixed + 'val_acc.npy', allow_pickle=True)


    empirical_risk_list_trainable = np.load(dir_trainable + 'empirical_risk.npy', allow_pickle=True)
    population_risk_list_trainable = np.load(dir_trainable + 'population_risk.npy', allow_pickle=True)
    complexity_list_trainable = np.load(dir_trainable + 'complexity.npy', allow_pickle=True)
    train_acc_list_trainable = np.load(dir_trainable + 'train_acc.npy', allow_pickle=True)
    val_acc_list_trainable = np.load(dir_trainable + 'val_acc.npy', allow_pickle=True)

    x = range(1, empirical_risk_list_fixed.size + 1)

    plt.figure(i_figure)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('generalization err v.s. comlexity {}'.format(tag), fontsize=16)
    plt.plot(x, train_acc_list_trainable, 'blue', label='Trainable_train', linewidth=2)
    plt.plot(x, train_acc_list_fixed, 'darkturquoise', label='Fixed_train', linestyle='--', linewidth=2)

    plt.plot(x, val_acc_list_trainable, 'tomato', label='Trainable_val', linestyle='--', linewidth=2)
    plt.plot(x, val_acc_list_fixed, 'gold', label='Fixed_val', linestyle='--', linewidth=2)

    plt.legend(fontsize=14, loc='best')

if __name__ == '__main__':
    # for 2 clients
    dir_fixed_2clients = '../results/medmnist_custom_0.5_fixed_ConvNet3_2clients_seed:1_2023-08-29-23:25:44/'
    # dir_trainable_2clients = '../results/medmnist_custom_0.5_trainable_ConvNet3_2clients_seed:2_2023-08-29-23:29:24/'
    dir_trainable_2clients ='../saved/medmnist_custom_0.5_trainable_OmConvNet_2clients_seed:1_2023-08-30-09:51:30/'
    my_plot(dir_fixed_2clients, dir_trainable_2clients, i_figure=1, tag='2clients')

    # for 10 clients
    dir_fixed_10clients = '../results/medmnist_custom_0.5_fixed_ConvNet3_10clients_seed:1_2023-08-29-23:38:23/'
    # dir_trainable_10clients = '../results/medmnist_custom_0.5_trainable_ConvNet3_10clients_seed:2_2023-08-29-23:55:09/'
    dir_trainable_10clients = '../saved/medmnist_custom_0.5_trainable_OmConvNet_10clients_seed:1_2023-08-30-09:51:46/'
    my_plot(dir_fixed_10clients, dir_trainable_10clients, i_figure=2, tag='10clients')

    plt.show()