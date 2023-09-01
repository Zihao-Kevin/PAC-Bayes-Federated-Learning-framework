import torch
from torch.autograd import Variable
# import data_gen
from .common import count_correct, get_value
from Models.stochastic_layers import StochasticLayer


def add_noise(param, std):
    return param + Variable(param.data.new(param.size()).normal_(0, std), requires_grad=False)


def add_noise_to_model(model, std):

    layers_list = [layer for layer in model.children() if isinstance(layer, StochasticLayer)]

    for i_layer, layer in enumerate(layers_list):
        if hasattr(layer, 'w'):
            add_noise(layer.w['log_var'], std)
            add_noise(layer.w['mean'], std)
        if hasattr(layer, 'b'):
            add_noise(layer.b['log_var'], std)
            add_noise(layer.b['mean'], std)


def kl_calculation(prm, post, prior, noised_prior=False):
    """KL divergence D_{KL}[post(x)||prior(x)] for a fully factorized Gaussian"""
    if noised_prior:
        prior_log_var = add_noise(prior['log_var'], prm.kappa_post)
        prior_mean = add_noise(prior['mean'], prm.kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)

    numerator = (post['mean'] - prior_mean).pow(2) + post_var
    denominator = prior_var
    kl = 1 / 2 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)

    # note: don't add small number to denominator, since we need to have zero KL when post==prior.

    return kl


def all_layer_kl_calculation(prm, prior_model, post_model, noised_prior=False):
    
    prior_layer_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLayer)]
    post_layer_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLayer)]

    total_kl = 0.0
    for i_layer, prior_layer in enumerate(prior_layer_list):
        post_layer = post_layer_list[i_layer]
        # if is weight:
        if hasattr(prior_layer, 'w'):
            total_kl += kl_calculation(prm, post_layer.w, prior_layer.w, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_kl += kl_calculation(prm, post_layer.b, prior_layer.b, noised_prior)

    return total_kl


def get_pac_bayes_bound(prm, prior_model, post_model, client_weight, n_train_clients, n_samples, llambda, noised_prior=False):
    complexity_type = prm.complexity_type
    #  confidence of the bound:
    all_layer_kl = all_layer_kl_calculation(prm, prior_model, post_model, noised_prior)

    if complexity_type == 'NoComplexity':
        # set as zero
        complex_term = Variable(cmn.zeros_gpu(1), requires_grad=False)

    elif complexity_type == 'PAC_Bayes_FL':
        # complex_term = 1 / llambda * total_kl + (llambda * C ** 2) / (8 * prm.n_train_clients * n_samples) + 1 / llambda * math.log(1 / delta)
        complex_term = client_weight * all_layer_kl / llambda
    else:
        raise ValueError('Invalid complexity_type')

    return complex_term

# ------------------------- Test code -------------------------
def run_test_Bayes(model, test_loader, loss_criterion, prm, verbose=1):

    if len(test_loader) == 0:
        return 0.0, 0.0

    if prm.test_type == 'MaxPosterior':
        info =  run_test_max_posterior(model, test_loader, loss_criterion, prm)
    elif prm.test_type == 'MajorityVote':
        info = run_test_majority_vote(model, test_loader, loss_criterion, prm, n_votes=5)
    elif prm.test_type == 'AvgVote':
        info = run_test_avg_vote(model, test_loader, loss_criterion, prm, n_votes=5)
    else:
        raise ValueError('Invalid test_type')
    if verbose:
        print('Test Accuracy: {:.3} ({}/{}), Test loss: {:.4}'.format(float(info['test_acc']), info['n_correct'],
                                                                      info['n_test_samples'], float(info['test_loss'])))
    return info['test_acc'], info['test_loss']


def run_test_max_posterior(model, test_loader, loss_criterion, prm):

    n_test_samples = len(test_loader.dataset)

    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)
        old_eps_std = model.set_eps_std(0.0)   # test with max-posterior
        outputs = model(inputs)
        model.set_eps_std(old_eps_std)  # return model to normal behaviour
        test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
        n_correct += count_correct(outputs, targets)

    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    info = {'test_acc':test_acc, 'n_correct':n_correct, 'test_type':'max_posterior',
            'n_test_samples':n_test_samples, 'test_loss':get_value(test_loss)}
    return info


def run_test_majority_vote(model, test_loader, loss_criterion, prm, n_votes=9):
    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)

        batch_size = inputs.shape[0] # min(prm.test_batch_size, n_test_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = cmn.zeros_gpu((batch_size, n_labels))
        for i_vote in range(n_votes):

            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets)
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max output
            for i_sample in range(batch_size):
                pred_val = pred[i_sample].cpu().numpy()[0]
                votes[i_sample, pred_val] += 1

        majority_pred = votes.max(1, keepdim=True)[1] # find argmax class for each sample
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()
    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    info = {'test_acc': test_acc, 'n_correct': n_correct, 'test_type': 'majority_vote',
            'n_test_samples': n_test_samples, 'test_loss': get_value(test_loss)}
    return info


def run_test_avg_vote(model, test_loader, loss_criterion, prm, n_votes=5):

    n_test_samples = len(test_loader.dataset)
    n_test_batches = len(test_loader)
    model.eval()
    test_loss = 0
    n_correct = 0
    for batch_data in test_loader:
        inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)

        batch_size = min(prm.test_batch_size, n_test_samples)
        info = data_gen.get_info(prm)
        n_labels = info['n_classes']
        votes = cmn.zeros_gpu((batch_size, n_labels))
        for i_vote in range(n_votes):

            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets)
            votes += outputs.data

        majority_pred = votes.max(1, keepdim=True)[1]
        n_correct += majority_pred.eq(targets.data.view_as(majority_pred)).cpu().sum()

    test_loss /= n_test_samples
    test_acc = n_correct / n_test_samples
    info = {'test_acc': test_acc, 'n_correct': n_correct, 'test_type': 'AvgVote',
            'n_test_samples': n_test_samples, 'test_loss': get_value(test_loss)}
    return info