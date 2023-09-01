echo "--------------------Start----------------"
cd ..

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 20 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 50 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 20 --dataset medmnist --partition_data custom --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-3

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 50 --dataset medmnist --partition_data custom --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-3
