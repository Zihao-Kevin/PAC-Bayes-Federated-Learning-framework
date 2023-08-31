echo "--------------------Start----------------"
cd ..
#
#python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4
#
#python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4
#
#python main.py --seed 3 --prior_type trainable --device cuda:1 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-1

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-2

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-1

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-2

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3


