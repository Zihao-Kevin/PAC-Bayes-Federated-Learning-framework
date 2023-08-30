import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .modelsel import modelsel
from .traineval import train, test
from .base_comm import communication
from .Fed_Utils.Bayes_utils import all_layer_kl_calculation
import math

class fedavg(torch.nn.Module):
    def __init__(self, prm):
        super(fedavg, self).__init__()
        self.prm = prm
        self.server_prior_model, self.server_post_model, self.client_prior_models, \
                self.client_post_models, self.client_weight = modelsel(prm, prm.device)

        if self.prm.prior_type == 'fixed':
            self.optimizers = [optim.SGD(params=self.client_post_models[idx].parameters(), lr=prm.lr) for idx in
                               range(prm.n_train_clients)]
        elif self.prm.prior_type == 'trainable':
            self.optimizers = [optim.SGD(params=list(self.client_prior_models[idx].parameters()) +
                            list(self.client_post_models[idx].parameters()), lr=prm.lr) for idx in range(prm.n_train_clients)]
        self.loss_fun = nn.CrossEntropyLoss(size_average=True).cuda()
        self.llambda = 0.
        self.additional_comp_term = 0.

    def client_train(self, c_idx, dataloader, round):
        if c_idx == 0:
            self.get_llambda()
            self.get_additional_comp()
        client_empirical_loss, client_complexity, correct_count, sample_count = train(
            self.prm, self.client_prior_models[c_idx], self.client_post_models[c_idx], self.client_weight[c_idx],
                    dataloader, self.optimizers[c_idx], self.loss_fun, self.llambda, self.prm.device)
        return client_empirical_loss, client_complexity, correct_count, sample_count

    def server_aggre(self):
        if self.prm.prior_type == 'trainable':
            self.client_prior_models = copy.deepcopy(self.client_post_models)
        self.server_post_model, self.client_post_models = communication(
            self.prm, self.server_post_model, self.client_post_models, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.prm, self.client_post_models[c_idx], dataloader, self.loss_fun, self.prm.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.prm, self.server_post_model, dataloader, self.loss_fun, self.prm.device)
        return train_loss, train_acc

    def get_llambda(self):
        total_kl = 0.
        for i in range(self.prm.n_train_clients):
            total_kl += all_layer_kl_calculation(self.prm, self.client_prior_models[i], self.client_post_models[i])
        self.llambda = math.sqrt(8 * self.prm.n_train_clients * self.prm.n_samples *
                            (total_kl + math.log(self.prm.n_samples / self.prm.delta)))

    def get_additional_comp(self):
        self.additional_comp_term = (self.llambda * (self.prm.C ** 2)) / (8 * self.prm.n_train_clients * self.prm.n_samples) \
                                    + 1 / self.llambda * math.log(1 / self.prm.delta)
