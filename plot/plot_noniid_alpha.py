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
    plt.plot(x, train_acc_list_trainable, 'blue', label='alpha{}_train'.format(tag[0]), linewidth=2)
    plt.plot(x, train_acc_list_fixed, 'darkturquoise', label='alpha{}_train'.format(tag[1]), linestyle='--', linewidth=2)

    plt.plot(x, val_acc_list_trainable, 'tomato', label='alpha{}_val'.format(tag[0]), linestyle='--', linewidth=2)
    plt.plot(x, val_acc_list_fixed, 'gold', label='alpha{}_val'.format(tag[1]), linestyle='--', linewidth=2)

    plt.legend(fontsize=14, loc='best')

if __name__ == '__main__':

    # for noniid 0.1 and 0.5
    dir_fixed_1 = '../saved/cifar10_non_iid_dirichlet_0.1_trainable_ConvNet4_10clients_seed:1_2023-08-30-11:56:09/'
    dir_trainable_5 = '../saved/cifar10_non_iid_dirichlet_0.5_trainable_ConvNet4_10clients_seed:1_2023-08-30-11:56:59/'
    my_plot(dir_fixed_1, dir_trainable_5, i_figure=3, tag=[0.1, 0.5])
    plt.show()