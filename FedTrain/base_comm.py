import torch
import copy


def communication(prm, server_post_model, client_post_models, client_weights):
    device = prm.device
    for key in server_post_model.state_dict().keys():
        if 'num_batches_tracked' in key:
            server_post_model.state_dict()[key].data.copy_(server_post_model.state_dict()[key])
        else:
            # First, aggregate the client models into the server model according to client weights:
            # $\mu = \sum_{k=1}^K p(k) \mu_k \sigma_k^{-2} / \sum_{k=1}^K p(k) \sigma_k^{-2}$
            # $\sigma = 1 / \sum_{k = 1}^K p(k) \sigma_k^{-2}$
            num_clients = len(client_weights)
            p_k = torch.tensor(client_weights)

            # Extract all keys for mu and log_var
            keys = list(server_post_model.state_dict().keys())
            mu_keys = [key for key in keys if 'mu' in key]
            log_var_keys = [key for key in keys if 'log_var' in key]

            for mu_key, log_var_key in zip(mu_keys, log_var_keys):

                mu_k = torch.stack([client_post_models[i].state_dict()[mu_key] for i in range(num_clients)])
                log_var_k = torch.stack(
                    [client_post_models[i].state_dict()[log_var_key] for i in range(num_clients)])
                sigma_k = torch.exp(log_var_k)

                # Compute aggregated mu and sigma
                p_k_adjusted = p_k.view(num_clients, *([1] * (mu_k.dim() - 1))).to(device)
                mu_numerator = torch.sum(p_k_adjusted * mu_k / sigma_k, dim=0)
                sigma_denominator = torch.sum(p_k_adjusted / sigma_k, dim=0)
                mu_global = mu_numerator / sigma_denominator
                sigma_square = 1 / sigma_denominator
                log_var_global = torch.log(sigma_square)

                # Update the server model
                server_post_model.state_dict()[mu_key].data.copy_(mu_global)
                server_post_model.state_dict()[log_var_key].data.copy_(log_var_global)

            # Broadcast the server model to each client
            for client_idx in range(num_clients):
                for mu_key, log_var_key in zip(mu_keys, log_var_keys):
                    client_post_models[client_idx].state_dict()[mu_key].data.copy_(
                        server_post_model.state_dict()[mu_key]
                    )
                    client_post_models[client_idx].state_dict()[log_var_key].data.copy_(
                        server_post_model.state_dict()[log_var_key]
                    )

    return server_post_model, client_post_models
    