echo "--------------------Start----------------"
cd ..

#python main.py --seed 1 --prior_type fixed --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom
#
#python main.py --seed 2 --prior_type fixed --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom
#
#python main.py --seed 3 --prior_type fixed --device cuda:2 --n_train_clients 2 --dataset medmnist --partition_data custom

python main.py --seed 1 --prior_type fixed --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data custom

python main.py --seed 2 --prior_type fixed --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data custom

python main.py --seed 3 --prior_type fixed --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data custom

python main.py --seed 1 --prior_type fixed --device cuda:2 --n_train_clients 20 --dataset medmnist --partition_data custom

python main.py --seed 2 --prior_type fixed --device cuda:2 --n_train_clients 20 --dataset medmnist --partition_data custom

python main.py --seed 3 --prior_type fixed --device cuda:2 --n_train_clients 20 --dataset medmnist --partition_data custom