import enum
import numpy as np
import torch


def evalandprint(prm, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed):
    # evaluation on training data
    train_acc_list = [None] * prm.n_train_clients
    train_loss_list = [None] * prm.n_train_clients
    for client_idx in range(prm.n_train_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        train_loss_list[client_idx] = train_loss.item()
        train_acc_list[client_idx] = train_acc
        print(
            f'Client-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    # evaluation on valid data
    val_loss_list = [None] * prm.n_train_clients
    val_acc_list = [None] * prm.n_train_clients
    for client_idx in range(prm.n_train_clients):
        val_loss, val_acc = algclass.client_eval(
            client_idx, val_loaders[client_idx])
        val_loss_list[client_idx] = val_loss.item()
        val_acc_list[client_idx] = val_acc
        print(
            f'Client-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(prm.n_train_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(prm.n_train_clients):
            _, test_acc = algclass.client_eval(
                client_idx, test_loaders[client_idx])
            print(
                f' TestClient-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f}')
            best_tacc[client_idx] = test_acc
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        for i, tmodel in enumerate(algclass.client_post_models):
            tosave['client_model_'+str(i)] = tmodel.state_dict()
        tosave['server_model'] = algclass.server_post_model.state_dict()
        torch.save(tosave, SAVE_PATH)

    return best_acc, best_tacc, best_changed, np.mean(train_loss_list), np.mean(val_loss_list), np.mean(np.mean(train_acc_list)), np.mean(val_acc_list)
