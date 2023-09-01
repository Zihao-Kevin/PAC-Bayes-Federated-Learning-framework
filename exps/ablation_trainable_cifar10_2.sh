echo "--------------------Start----------------"
cd ..

#python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom
#
#python main.py --seed 2 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom
#
#python main.py --seed 3 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4

python main.py --seed 2 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4

python main.py --seed 3 --prior_type trainable --device cuda:2 --n_train_clients 2 --dataset cifar10 --partition_data custom --model_name ConvNet4
