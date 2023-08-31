import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

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
            self.optimizers = [optim.Adam(params=self.client_post_models[idx].parameters(), lr=prm.lr, weight_decay=1e-4) for idx in
                               range(prm.n_train_clients)]
        elif self.prm.prior_type == 'trainable':
            self.optimizers = [optim.Adam(params=list(self.client_prior_models[idx].parameters()) +
                            list(self.client_post_models[idx].parameters()), lr=prm.lr, weight_decay=5e-4) for idx in range(prm.n_train_clients)]
        self.schedulers = [ExponentialLR(self.optimizers[idx], gamma=0.98, last_epoch=-1) for idx in range(prm.n_train_clients)]
        self.loss_fun = nn.CrossEntropyLoss(size_average=True).cuda()
        self.weighted_total_kl = 0.
        self.llambda = 0.
        self.epoch_comp_term = 0.

    def client_train(self, c_idx, dataloader, round):
        if c_idx == 0:
            self.get_weighted_llambda()
            self.get_llambda()
            self.get_additional_comp()
        client_empirical_loss, client_complexity, correct_count, sample_count = train(
            self.prm, self.client_prior_models[c_idx], self.client_post_models[c_idx], self.client_weight[c_idx],
                    dataloader, self.optimizers[c_idx], self.schedulers[c_idx], self.loss_fun, self.llambda, self.prm.device)
        return client_empirical_loss, client_complexity, correct_count, sample_count

    def server_aggre(self):
        if self.prm.prior_type == 'trainable':
            self.client_prior_models = copy.deepcopy(self.client_post_models)
        for c_idx in range(self.prm.n_train_clients):
            self.schedulers[c_idx].step()
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

    def get_weighted_llambda(self):
        total_kl = 0.
        for i in range(self.prm.n_train_clients):
            total_kl += self.client_weight[i] * all_layer_kl_calculation(self.prm, self.client_prior_models[i],
                                                                         self.client_post_models[i])
        self.weighted_total_kl = total_kl

    def get_llambda(self):
        self.llambda = math.sqrt(8 * self.prm.n_train_clients * self.prm.n_samples *
                            (self.weighted_total_kl + math.log(self.prm.n_samples / self.prm.delta))) / self.prm.C

    def get_additional_comp(self):
        print(f"weighted_total_kl = {self.weighted_total_kl }")
        self.epoch_comp_term = self.prm.C * math.sqrt((self.weighted_total_kl + math.log(self.prm.n_samples / self.prm.delta))
                                                      / (2 * self.prm.n_train_clients * self.prm.n_samples))