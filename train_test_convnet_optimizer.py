# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from test_convnet_optimizer import TestConvNet
from common.trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

for optimizer in ['Adam', 'SGD', 'Adagrad', 'Momentum']:
    network = TestConvNet()
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                           epochs=max_epochs, mini_batch_size=100,
                           optimizer=optimizer, optimizer_param={'lr': 0.001},
                           evaluate_sample_num_per_epoch=1000)
    trainer.train()
    network.save_params("test_convnet_" + optimizer.lower() + "_params.pkl")
    with open("trainer_" + optimizer.lower() + ".pkl", 'wb') as f:
        pickle.dump(trainer, f)

print("Saved Network Parameters!")