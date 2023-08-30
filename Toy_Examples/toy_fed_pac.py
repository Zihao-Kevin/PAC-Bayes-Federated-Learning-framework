
from __future__ import absolute_import, division, print_function

import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from Utils.common import get_value
import torch
from torch.autograd import Variable
import torch.optim as optim
import math


def kl_cal(w_P_mu, w_P_log_sigma, w_mu, w_log_sigma, i_client):
    sigma_sqr_prior = torch.exp(2 * w_P_log_sigma[i_client])
    sigma_sqr_post = torch.exp(2 * w_log_sigma[i_client])
    return torch.sum(w_P_log_sigma[i_client] - w_log_sigma[i_client] +
              ((w_mu[i_client] - w_P_mu[i_client]).pow(2) + sigma_sqr_post[i_client]) /
              (2 * sigma_sqr_prior[i_client]) - 0.5)


def get_llambda(w_P_mu, w_P_log_sigma, w_mu, w_log_sigma, n_clients, n_samples):
    delta = 0.95
    total_kl_dist = 0
    for i in range(n_clients):
        kl_dist = kl_cal(w_P_mu, w_P_log_sigma, w_mu, w_log_sigma, i)
        if kl_dist < 0:
            kl_dist = 0
        total_kl_dist += kl_dist
    llambda = math.sqrt(8 * n_clients * n_samples * (total_kl_dist + math.log(4 * n_clients * n_samples / delta)))
    return llambda


def learn(data_set, complexity_type):

    n_clients = len(data_set)
    n_dim = data_set[0].shape[1]
    n_samples_list = [client_data.shape[0] for client_data in data_set]
    total_samples = np.sum(np.array(n_samples_list))

    # Init posteriors:
    w_mu = [Variable(torch.randn(n_dim).cuda(), requires_grad=True) for _ in range(n_clients)]
    w_log_sigma = [Variable(torch.randn(n_dim).cuda(), requires_grad=True) for _ in range(n_clients)]

    # Define prior:
    w_P_mu = copy.deepcopy(w_mu)
    w_P_log_sigma = copy.deepcopy(w_log_sigma)

    learning_rate = 1e-1

    n_epochs = 100
    batch_size = 128
    # Complexity terms:
    complex_term_sum = 0

    for i_epoch in range(n_epochs):

        llambda = get_llambda(w_P_mu, w_P_log_sigma, w_mu, w_log_sigma, n_clients, total_samples)
        print(f"llambda = {llambda}")

        for i_client in range(n_clients):

            optimizer = optim.Adam([w_mu[i_client], w_log_sigma[i_client], w_P_mu[i_client], w_P_log_sigma[i_client]], lr=learning_rate)

            # Sample data batch:
            batch_size_curr = min(n_samples_list[i_client], batch_size)
            batch_inds = np.random.choice(n_samples_list[i_client], batch_size_curr, replace=False)
            client_data = torch.from_numpy(data_set[i_client][batch_inds])
            client_data = Variable(client_data.cuda(), requires_grad=False)

            # Re-Parametrization:
            w_sigma = torch.exp(w_log_sigma[i_client])
            epsilon = Variable(torch.randn(n_dim).cuda(), requires_grad=False)
            w = w_mu[i_client] + w_sigma * epsilon

            # Empirical Loss:
            empirical_loss = (w - client_data).pow(2).mean()
            # empirical_loss_2 = empirical_loss * llambda

            n_samples = n_samples_list[i_client]

            kl_dist = kl_cal(w_P_mu, w_P_log_sigma, w_mu, w_log_sigma, i_client)
            complex_term = (kl_dist * n_samples) / (llambda * total_samples)

            if complexity_type == 'PAC_Bayes_McAllaster':
                delta = 0.95
                # complex_term_sum += torch.sqrt((1 / (2 * n_samples)) *
                #                            (kl_dist + np.log(2 * np.sqrt(n_samples) / delta)))
                complex_term_sum += kl_dist / llambda + np.log(2 * np.sqrt(n_samples) / delta)

            elif complexity_type == 'Variational_Bayes':
                complex_term_sum += (1 / n_samples) * kl_dist

            elif complexity_type == 'KL':
                complex_term_sum += kl_dist
            else:
                raise ValueError('Invalid complexity_type')

            objective = empirical_loss + complex_term
            # Gradient step:
            optimizer.zero_grad()  # zero the gradient buffers
            objective.backward(retain_graph=True)
            optimizer.step()  # Does the update
            print(f"empirical_loss = {empirical_loss}, complex_term = {complex_term}")

        if i_epoch % 100 == 0:
            print('Step: {0}, objective: {1}'.format(i_epoch, get_value(objective)))

    # Switch  back to numpy:
    w_mu = [w_mu[i].data.cpu().numpy() for i in range(n_clients)]
    w_sigma = [np.exp(w_log_sigma[i].data.cpu().numpy()) for i in range(n_clients)]
    w_P_mu = [w_P_mu[i].data.cpu().numpy() for i in range(n_clients)]
    w_P_sigma = [np.exp(w_P_log_sigma[i].data.cpu().numpy()) for i in range(n_clients)]

    print(f"w_mu = {w_mu}, w_sigma = {w_sigma}, w_P_mu = {w_P_mu}, w_P_sigma = {w_P_sigma}")

    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    for i_client in range(n_clients):
        # plot prior:
        plt.plot(w_P_mu[i_client][0], w_P_mu[i_client][1], 'o', label='prior {} mean'.format(1+i_client))
        ell = Ellipse(xy=(w_P_mu[i_client][0], w_P_mu[i_client][1]),
                      width=w_P_sigma[i_client][0], height=w_P_sigma[i_client][1],
                      angle=0, color='blue')
        ell.set_facecolor('none')
        ax.add_artist(ell)

        # plot client data points:
        plt.plot(data_set[i_client][:, 0], data_set[i_client][:, 1], '.',
                 label='client {0} samples'.format(1+i_client))
        # plot posterior:
        plt.plot(w_mu[i_client][0], w_mu[i_client][1], 'o', label='posterior {0} mean'.format(1+i_client))
        ell = Ellipse(xy=(w_mu[i_client][0], w_mu[i_client][1]),
                      width=w_sigma[i_client][0], height=w_sigma[i_client][1],
                      angle=0, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)

    # plt.plot(0, 0, 'x', label='hyper-prior ')

    # plt.legend(loc='upper left')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.xticks( np.arange(1, 5) )
    # plt.yticks( np.arange(0, 2, 0.5) )

    plt.savefig('ToyFig1.pdf', format='pdf', bbox_inches='tight')

    plt.show()