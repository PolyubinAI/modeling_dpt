install:
	pip install -r requirements.txt && pip install lightning[extra]
	pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

train:
	python3 /home/apolyubin/private_data/modeling-dpt/src/train.py -1

train_optuna:
	python3 /home/apolyubin/private_data/modeling-dpt/src/train_optuna.py -1
