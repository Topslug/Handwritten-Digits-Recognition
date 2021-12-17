# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from test_convnet_dropout import TestConvNet, TestConvNet_Dropout
from common.trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = TestConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
network.save_params("test_convnet_no_dropout_params.pkl")
with open("trainer_no_dropout.pkl", 'wb') as f:
    pickle.dump(trainer, f)

network_dropout = TestConvNet_Dropout()
trainer_dropout = Trainer(network_dropout, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer_dropout.train()

# 매개변수 보관
network.save_params("test_convnet_dropout_params.pkl")
with open("trainer_dropout.pkl", 'wb') as f:
    pickle.dump(trainer_dropout, f)

print("Saved Network Parameters!")

# 그래프 그리기
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(max_epochs)
# plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
# plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
#
# plt.plot(trainer.train_loss_list)
# plt.show()