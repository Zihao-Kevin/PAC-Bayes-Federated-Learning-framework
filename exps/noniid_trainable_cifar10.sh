echo "--------------------Start----------------"
cd ..

python main.py --seed 1 --prior_type trainable --device cuda:1 -- --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 3 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.2 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.3 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.4 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.5 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.6 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.7 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.8 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data non_iid_dirichlet --non_iid_alpha 0.9 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data custom --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data custom --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 3 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data custom --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data evenly --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 2 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data evenly --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3

python main.py --seed 3 --prior_type trainable --device cuda:1 --n_train_clients 10 --dataset cifar10 --partition_data evenly --non_iid_alpha 0.1 --model_name ConvNet4 --lr 1e-3
