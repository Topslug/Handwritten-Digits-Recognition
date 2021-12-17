# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
from test_convnet_pooling import TestConvNet, TestConvNet_Pooling
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

name_list = ["no_pooling", "pooling"]#no_pooling : 1,2층에 pooling 없음, pooling : 1,2층에 pooling 있음 - 마지막 층에는 둘다 pooling 있음
network = TestConvNet_Pooling()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
trainer.train()
network.save_params("test_convnet_" + name_list[1] + "_params.pkl")
with open("trainer_" + name_list[1] + ".pkl", 'wb') as f:
    pickle.dump(trainer, f)

# for i, network in enumerate([TestConvNet(), TestConvNet_Pooling()]):
#     trainer = Trainer(network, x_train, t_train, x_test, t_test,
#                       epochs=max_epochs, mini_batch_size=100,
#                       optimizer='Adam', optimizer_param={'lr': 0.001},
#                       evaluate_sample_num_per_epoch=1000)
#     trainer.train()
#     network.save_params("test_convnet_" + name_list[i] + "_params.pkl")
#     with open("trainer_" + name_list[i] + ".pkl", 'wb') as f:
#         pickle.dump(trainer, f)

print("Saved Network Parameters!")