echo "--------------------Start----------------"
cd ..
python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-1

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.2 --model_name ConvNet3 --lr 1e-2

python main.py --seed 1 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.2 --model_name ConvNet3 --lr 1e-3

python main.py --seed 2 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --model_name ConvNet3 --lr 1e-1

python main.py --seed 2 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.2 --model_name ConvNet3 --lr 1e-2

python main.py --seed 2 --prior_type trainable --device cuda:2 --n_train_clients 10 --dataset medmnist --partition_data non_iid_dirichlet --non_iid_alpha 0.2 --model_name ConvNet3 --lr 1e-3