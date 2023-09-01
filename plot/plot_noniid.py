import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glob
import os

# dataset = 'medmnist'
dataset = 'cifar10'
plt.figure(1)
colors = plt.cm.tab10.colors
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('different dirichlent alpha for {}'.format(dataset), fontsize=16)

folder_path = '../results/{}/noniid'.format(dataset)
all_files = glob.glob(os.path.join(folder_path, '*'))
train_acc = {}
val_acc = {}
comp = {}
for file in all_files:
    file_name = file.split('/')[-1].split('_')
    if file_name[8] == 'seed:1':
        alpha = float(file_name[4])
        empirical_risk_list = np.load(file + '/empirical_risk.npy', allow_pickle=True)[2:]
        population_risk_list = np.load(file + '/population_risk.npy', allow_pickle=True)[2:]
        complexity_list = np.load(file + '/complexity.npy', allow_pickle=True)[2:]
        train_acc_list = np.load(file + '/train_acc.npy', allow_pickle=True)[2:]
        val_acc_list = np.load(file + '/val_acc.npy', allow_pickle=True)[2:]
        train_acc[alpha] = train_acc_list[-1]
        val_acc[alpha] = val_acc_list[-1]
        comp[alpha] = complexity_list[-1]

sorted_train_acc = {k: train_acc[k] for k in sorted(train_acc)}
plt.plot(list(sorted_train_acc.keys()), list(sorted_train_acc.values()), c='r', label="train accuracy", linewidth=1)

sorted_val_acc = {k: val_acc[k] for k in sorted(val_acc)}
plt.plot(list(sorted_val_acc.keys()), list(sorted_val_acc.values()), c='b', label="val accuracy", linewidth=1)

sorted_comp = {k: comp[k] for k in sorted(comp)}
plt.plot(list(sorted_comp.keys()), list(sorted_comp.values()), c='g', label="complexity", linewidth=1)

# plt.plot(list(sorted_val_acc.keys()), [val_acc_value - train_acc_value for val_acc_value, train_acc_value in zip(sorted_train_acc.values(), sorted_val_acc.values())], c='gold', label="generalization error", linewidth=1)

#     x = range(1, res[alpha].size + 1)
#     plt.plot(x, res[alpha], c=colors[int(float(alpha)*10)], label='alpha={}'.format(float(alpha)), linewidth=1)
plt.legend(fontsize=12, loc='best')
plt.show()