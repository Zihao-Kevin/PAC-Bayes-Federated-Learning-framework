from Fed_Utils import data_gen
from Fed_Utils.common import count_correct, get_value
from Fed_Utils import Bayes_utils

def get_objective(prm, prior_model, mb_data_loaders, mb_iterators, mb_posteriors_models, loss_criterion, n_train_clients):
    n_clients_in_mb = len(mb_data_loaders)

    sum_empirical_loss = 0
    sum_intra_task_comp = 0
    correct_count = 0
    sample_count = 0

    # ----------- loop over all clients in federated learning -----------------------------------#
    for i_client in range(n_clients_in_mb):
        
        n_samples = mb_data_loaders[i_client]['n_train_samples']

        # get sample-batch data from current task to calculate the empirical loss estimate:
        batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_client], mb_data_loaders[i_client]['train'])

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_client]
        post_model.train()

        # Monte-Carlo iterations:
        n_MC = prm.n_MC
        client_empirical_loss = 0
        client_complexity = 0
        #  ----------------------------------- Monte-Carlo loop  -----------------------------------#
        for i_MC in range(n_MC):
            # get batch variables:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Empirical Loss on current client:
            outputs = post_model(inputs)
            curr_empirical_loss = loss_criterion(outputs, targets)

            correct_count += count_correct(outputs, targets)
            sample_count += input.size(0)

            # Intra-client complexity of current client
            curr_empirical_loss, curr_complexity = Bayes_utils.get_pac_bayes_bound(
                prm, prior_model, post_model, n_samples, curr_empirical_loss, 
                n_train_clients, noised_prior=True
            )
            
            #  ??????
            client_empirical_loss += (1 / n_MC) * curr_empirical_loss
            client_complexity += (1 / n_MC) * curr_complexity

        sum_empirical_loss += client_empirical_loss
        sum_intra_task_comp += client_complexity
    
    avg_empirical_loss = (1 / n_clients_in_mb) * sum_empirical_loss
    avg_intra_task_comp = (1 / n_clients_in_mb) * sum_intra_task_comp

    # Approximated total objective:
    total_objective = avg_empirical_loss + avg_intra_task_comp

    info = {'sample_count': get_value(sample_count), 'correct_count': get_value(correct_count),
            'avg_empirical_loss': get_value(avg_empirical_loss),
            'avg_intra_task_comp': get_value(avg_intra_task_comp),
            }
    return total_objective, info

