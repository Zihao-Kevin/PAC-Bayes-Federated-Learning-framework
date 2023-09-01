import torch
import torch.nn as nn
import torch.optim as optim

from util.modelsel import modelsel
from util.traineval import train, test
from .base_comm import communication


class fedavg(torch.nn.Module):
    def __init__(self, prm):
        super(fedavg, self).__init__()
        self.server_model, self.client_model, self.client_weight = modelsel(
            prm, prm.device)
        self.optimizers = [optim.Adam(params=self.client_model[idx].parameters(
        ), lr=prm.lr) for idx in range(prm.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.prm = prm

    def client_train(self, c_idx, dataloader, round):
        train_loss, train_acc = train(
            self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.prm.device)
        return train_loss, train_acc

    def server_aggre(self):
        self.server_model, self.client_model = communication(
            self.prm, self.server_model, self.client_model, self.client_weight)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = test(
            self.client_model[c_idx], dataloader, self.loss_fun, self.prm.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.prm.device)
        return train_loss, train_acc
