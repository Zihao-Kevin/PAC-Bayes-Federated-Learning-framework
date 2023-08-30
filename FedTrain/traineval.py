import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from .datasplit import define_pretrain_dataset
from .prepare_data import get_whole_dataset

from .Fed_Utils.common import count_correct
from .Fed_Utils import Bayes_utils
import copy

def train(prm, prior_model, post_model, client_weight, data_loader, optimizer, loss_criterion, llambda, device):

    post_model.train()
    if prm.prior_type == 'fixed':
        prior_model.eval()
    elif prm.prior_type == 'trainable':
        prior_model.train()

    # Monte-Carlo iterations:
    n_MC = prm.n_MC
    client_empirical_loss = 0.
    client_complexity = 0.
    correct_count = 0
    sample_count = 0
    cnt = 0
    #  ----------------------------------- Monte-Carlo loop  -----------------------------------#
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        # Empirical Loss on current client:
        outputs = post_model(inputs)
        curr_empirical_loss = loss_criterion(outputs, targets)

        sample_count += inputs.size(0)
        pred = outputs.data.max(1)[1]
        correct_count += pred.eq(targets.view(-1)).sum().item()

        # Intra-client complexity of current client
        curr_complexity = Bayes_utils.get_pac_bayes_bound(
            prm, prior_model, post_model, client_weight, prm.n_train_clients, prm.n_samples, llambda, noised_prior=False
        )
        
        client_empirical_loss += curr_empirical_loss / n_MC
        client_complexity += curr_complexity / n_MC

        cnt += 1
        if cnt >= n_MC: # equivalent to n times MCMC
            break

    total_local_objective = client_empirical_loss + client_complexity

    optimizer.zero_grad()
    total_local_objective.backward()
    # torch.nn.utils.clip_grad_norm(parameters, 0.25)
    optimizer.step()

    return client_empirical_loss, client_complexity, correct_count, sample_count


def test(prm, model, test_loader, loss_criterion, device):
    model.eval()
    n_test_samples = len(test_loader.dataset)
    test_loss = 0.
    n_correct = 0.
    total_samples = 0.
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        old_eps_std = model.set_eps_std(0.0)   # test with max-posterior
        outputs = model(inputs)
        model.set_eps_std(old_eps_std)  # return model to normal behaviour
        test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)
        total_samples += targets.size(0)

    test_loss /= len(test_loader)
    test_acc = n_correct / total_samples

    # info = {'test_acc':test_acc, 'n_correct':n_correct, 'test_type':'max_posterior',
    #         'n_test_samples':n_test_samples, 'test_loss':get_value(test_loss)}
    # print('Test Accuracy: {:.3} ({}/{}), Test loss: {:.4}'.format(float(info['test_acc']), info['n_correct'],
    #                                     info['n_test_samples'], float(info['test_loss'])))
    return test_loss, test_acc


def get_value(x):
    ''' Returns the value of any scalar type'''
    if isinstance(x, Variable):
        if hasattr(x, 'item'):
            return x.item()
        else:
            return x.data[0]
    else:
        return x

# def test(model, data_loader, loss_fun, device):
#     model.eval()
#     loss_all = 0
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in data_loader:
#             data = data.to(device).float()
#             target = target.to(device).long()
#             output = model(data)
#             loss = loss_fun(output, target)
#             loss_all += loss.item()
#             total += target.size(0)
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.view(-1)).sum().item()
#
#         return loss_all / len(data_loader), correct/total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam*F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    for _ in range(args.pretrained_iters):
        _, acc = train(model, traindata, opt, loss_fun, device)
    torch.save({
        'state': model.state_dict(),
        'acc': acc
    }, filename)
    print('===done!===')
