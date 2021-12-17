# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from test_convnet_initialize import TestConvNet_Relu, TestConvNet_Sigmoid
from common.trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network_relu = TestConvNet_Relu(initialize='He')
trainer_relu = Trainer(network_relu, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_relu.train()
network_relu.save_params("test_convnet_relu_he_params.pkl")
with open("trainer_relu_he.pkl", 'wb') as f:
    pickle.dump(trainer_relu, f)

network_sigmoid = TestConvNet_Sigmoid(initialize='He')
trainer_sigmoid = Trainer(network_sigmoid, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_sigmoid.train()
network_sigmoid.save_params("test_convnet_sigmoid_he_params.pkl")
with open("trainer_sigmoid_he.pkl", 'wb') as f:
    pickle.dump(trainer_sigmoid, f)

network_relu = TestConvNet_Relu(initialize='Xavier')
trainer_relu = Trainer(network_relu, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_relu.train()
network_relu.save_params("test_convnet_relu_xavier_params.pkl")
with open("trainer_relu_xavier.pkl", 'wb') as f:
    pickle.dump(trainer_relu, f)

network_sigmoid = TestConvNet_Sigmoid(initialize='Xavier')
trainer_sigmoid = Trainer(network_sigmoid, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_sigmoid.train()
network_sigmoid.save_params("test_convnet_sigmoid_xavier_params.pkl")
with open("trainer_sigmoid_xavier.pkl", 'wb') as f:
    pickle.dump(trainer_sigmoid, f)

network_relu = TestConvNet_Relu(initialize=0.01)
trainer_relu = Trainer(network_relu, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_relu.train()
network_relu.save_params("test_convnet_relu_0.01_params.pkl")
with open("trainer_relu_0.01.pkl", 'wb') as f:
    pickle.dump(trainer_relu, f)

network_sigmoid = TestConvNet_Sigmoid(initialize=0.01)
trainer_sigmoid = Trainer(network_sigmoid, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_sigmoid.train()
network_sigmoid.save_params("test_convnet_sigmoid_0.01_params.pkl")
with open("trainer_sigmoid_0.01.pkl", 'wb') as f:
    pickle.dump(trainer_sigmoid, f)

print("Saved Network Parameters!")