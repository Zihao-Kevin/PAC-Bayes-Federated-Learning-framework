
from __future__ import absolute_import, division, print_function

import argparse
import timeit, os
# sys.path.append(os.path.abspath(os.path.join('./FedTrain')))
# sys.path.append(os.path.abspath(os.path.join('../Models')))
# sys.path.append(os.path.abspath(os.path.join('./FedTrain/Fed_Utils')))

import numpy as np
import torch
import torch.optim as optim

from Models.stochastic_models import get_model
from FedTrain.Fed_Utils.common import load_model_state, create_result_dir, set_random_seed, write_to_log

from FedTrain.prepare_data import get_data
from FedTrain.fedavg import fedavg
from FedTrain.evalandprint import evalandprint
from FedTrain.Fed_Utils.common import get_data_path, img_param_init
torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ---------------------- Run Parameters ----------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--mode', type=str, help='FederatedTrain or LoadFederatedTrain',
                    default='FederatedTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--load_model_path', type=str, help='set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)',
                    default='')

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=10)

parser.add_argument('--datapercent', type=float,
                    default=0.7, help='data percent to use')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='[vlcs | pacs | officehome | pamap | covid | medmnist | cifar10]')

parser.add_argument('--model_name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet4')  # ConvNet4 / 'FcNet3' / 'ConvNet3'

parser.add_argument('--root_dir', type=str,
                default='/home/zihao/PycharmProjects/PAC-Bayes-Federated-Learning-framework/data/', help='data path')

parser.add_argument('--save_path', type=str,
                    default='./cks/', help='path to save the checkpoint')

parser.add_argument('--device', type=str,
                    default='cuda', help='[cuda | cpu]')

# ---------------------- FL Parameters ----------------------#

parser.add_argument('--n_train_clients', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=20)

parser.add_argument('--non_iid_alpha', type=float,
                        default=0.5, help='data split for label shift')

parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way, non_iid_dirichlet or custom')

parser.add_argument('--partition_ratios', type=list, help='Data partition ratio, [0.1, 0.2, 0.3, 0.4]',
                    default=[0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2])

parser.add_argument('--local_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=20)  #

parser.add_argument('--global_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=300)  #

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='Permute_Labels')

# ---------------------- Algorithm Parameters ----------------------#

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='PAC_Bayes_FL')  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--prior_type', type=str,
                    help="fixed prior or trainable prior",
                    default='fixed')

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-2)

parser.add_argument('--batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=128)
# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.data_path = get_data_path()
prm.random_state = np.random.RandomState(1)
set_random_seed(prm.seed)
create_result_dir(prm)

if prm.dataset in ['vlcs', 'pacs', 'off_home']:
    prm = img_param_init(prm)
    prm.n_train_clients = 4
    prm.local_steps = 1
    prm.lr = 1e-3

if prm.n_train_clients == 10:
    prm.partition_ratios = [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]
elif prm.n_train_clients == 2:
    prm.partition_ratios = [0.2, 0.8]
elif prm.n_train_clients == 4:
    prm.partition_ratios = [0.15, 0.2, 0.25, 0.4]
prm.partition_data_origin = prm.partition_data
# Weights initialization (for Bayesian net):
prm.log_var_init = {'mean': -10, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9, 'weight_decay': 1e-4}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay

# MPB alg params:
prm.delta = 0.95  #  maximal probability that the bound does not hold
prm.C = 1
prm.n_samples = 0.
empirical_risk_list = []
population_risk_list = []
complexity_list = []
train_acc_list = []
val_acc_list = []
# init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# path to save the learned meta-parameters
save_path = os.path.join(prm.result_dir, 'model.pt')
emp_save_path = os.path.join(prm.result_dir, 'empirical_risk.npy')
pop_save_path = os.path.join(prm.result_dir, 'population_risk.npy')
comp_save_path = os.path.join(prm.result_dir, 'complexity.npy')
train_acc_save_path = os.path.join(prm.result_dir, 'train_acc.npy')
val_acc_save_path = os.path.join(prm.result_dir, 'val_acc.npy')
# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

if prm.mode == 'FederatedTrain':
    best_changed = False
    n_train_clients = prm.n_train_clients
    best_acc = [0] * n_train_clients
    best_tacc = [0] * n_train_clients
    start_iter = 0
    if n_train_clients:
        # In this case we generate a finite set of train (observed) task before meta-training.
        # Generate the data sets of the training tasks:
        write_to_log('--- Generating {} training-tasks'.format(n_train_clients), prm)
        train_loaders, val_loaders, test_loaders = get_data(prm.dataset)(prm)
        if prm.partition_data_origin == 'non_iid_dirichlet':
            prm.partition_ratios = [len(train_loaders[i].dataset.indices) for i in range(n_train_clients)]
            prm.partition_ratios /= np.sum(prm.partition_ratios)

        fed_training = fedavg(prm)

        # global training
        for g_iter in range(prm.global_steps):
            print(f"============ Train round {g_iter} ============")
            sum_empirical_loss = 0
            sum_total_comp = 0
            for wi in range(prm.local_steps):
                for client_idx in range(prm.n_train_clients):
                    client_empirical_loss, client_complexity, correct_count, sample_count = \
                        fed_training.client_train(client_idx, train_loaders[client_idx], g_iter)

                    sum_empirical_loss += client_empirical_loss * prm.partition_ratios[client_idx]
                    sum_total_comp += client_complexity
                sum_total_comp += fed_training.additional_comp_term

            avg_empirical_loss = sum_empirical_loss / prm.local_steps
            avg_total_comp = sum_total_comp / prm.local_steps
            total_objective = avg_empirical_loss + avg_total_comp

            complexity_list.append(avg_total_comp.item())
            # server aggregation
            fed_training.server_aggre()

            best_acc, best_tacc, best_changed, empirical_risk, population_risk, train_acc, val_acc = evalandprint(
                prm, fed_training, train_loaders, val_loaders, test_loaders, save_path, best_acc, best_tacc, g_iter, best_changed
            )

            print(f"empirical_risk = {empirical_risk}, avg_empirical_loss = {avg_empirical_loss}")
            empirical_risk_list.append(empirical_risk)
            population_risk_list.append(population_risk)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

        np.save(comp_save_path, np.array(complexity_list))
        np.save(emp_save_path, np.array(empirical_risk_list))
        np.save(pop_save_path, np.array(population_risk_list))
        np.save(train_acc_save_path, np.array(train_acc_list))
        np.save(val_acc_save_path, np.array(val_acc_list))

        write_to_log('Trained prior saved in ' + save_path, prm)
    else:
        # In this case we observe new tasks generated from the task-distribution in each meta-iteration.
        write_to_log('---- Infinite train tasks - New training tasks are '
                     'drawn from tasks distribution in each iteration...', prm)

        # Meta-training to learn meta-prior (theta params):
        # prior_model = meta_train_Bayes_infinite_tasks.run_meta_learning(task_generator, prm)
elif prm.mode == 'LoadFederatedTrain':
    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    # Then load the weights:
    load_model_state(prior_model, prm.load_model_path)
    write_to_log('Pre-trained  prior loaded from ' + prm.load_model_path, prm)
else:
    raise ValueError('Invalid mode')
# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
#  Print prior analysis
# run_prior_analysis(prior_model)

# stop_time = timeit.default_timer()
# write_to_log('Total runtime: ' +
#              time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm)
#
# #  Print results
# write_to_log('----- Final Results: ', prm)
# write_to_log('----- Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
#              .format(100 * test_err_vec.mean(), 100 * test_err_vec.std()), prm)

# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------
# from Single_Task import learn_single_standard
# test_err_standard = np.zeros(n_test_tasks)
# for i_task in range(n_test_tasks):
#     print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
#     task_data = test_tasks_data[i_task]
#     test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm, verbose=0)
#
# write_to_log('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
#              format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm)
