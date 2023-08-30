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
    empirical_risk_list = np.load(dir + 'empirical_risk.npy', allow_pickle=True)
    population_risk_list = np.load(dir + 'population_risk.npy', allow_pickle=True)
    complexity_list = np.load(dir + 'complexity.npy', allow_pickle=True)
    train_acc_list = np.load(dir + 'train_acc.npy', allow_pickle=True)
    val_acc_list = np.load(dir + 'val_acc.npy', allow_pickle=True)

    x = range(1, empirical_risk_list.size + 1)

    plt.figure(1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('{} Parameters'.format('generalization err v.s. comlexity'), fontsize=16)
    plt.plot(x, population_risk_list - empirical_risk_list, 'darkturquoise', label='empirical_risk_list',linestyle='--', linewidth=2)
    plt.plot(x, complexity_list, 'blue', label='complexity_list', linewidth=2)
    plt.legend(fontsize=14, loc='best')

    plt.figure(2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('{} Parameters'.format('train acc v.s. val acc'), fontsize=16)
    plt.plot(x, train_acc_list, 'darkturquoise', label='train acc',linestyle='--', linewidth=2)
    plt.plot(x, val_acc_list, 'blue', label='val acc', linewidth=2)
    plt.legend(fontsize=14, loc='best')

if __name__ == '__main__':
    dir = '../saved/medmnist_custom_fixed_4clients_2023-08-29-13:47:52/'
    plt.show()