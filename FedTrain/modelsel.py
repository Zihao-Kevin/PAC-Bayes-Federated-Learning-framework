import copy
from Models.stochastic_models import get_model


def modelsel(prm, device):
    # if prm.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
    #     server_model = AlexNet(num_classes=prm.num_classes).to(device)
    # elif 'medmnist' in prm.dataset:
    #     server_model = lenet5v().to(device)
    # elif 'pamap' in prm.dataset:
    #     server_model = PamapModel().to(device)

    server_prior_model = get_model(prm)
    server_post_model = get_model(prm)

    client_weights = prm.partition_ratios
    client_post_models = [copy.deepcopy(server_post_model).to(device)
              for _ in range(prm.n_train_clients)]
    client_prior_models = [copy.deepcopy(server_post_model).to(device)
              for _ in range(prm.n_train_clients)]
    return server_prior_model, server_post_model, client_prior_models, client_post_models, client_weights
