#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
import chainer
import numpy as np
import mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Sequential(
            L.Linear(input_size, hidden_size), F.relu,
            L.Linear(hidden_size, output_size)
        )
    
    def predict(self, x):
        return self.net(x)

    def loss(self, x, t):
        y = self.predict(x)
        return F.softmax_cross_entropy(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        accuracy = F.accuracy(y, t)
        return accuracy

    def gradient(self, x, t):
        loss = self.loss(x, t)
        self.net.cleargrads()
        loss.backward()
        return loss.grad
    

if __name__ == "__main__":
    dataset = mnist.load_dataset()
    x_train = dataset['train_img']
    t_train = dataset['train_label']

    x_test = dataset['test_img']
    t_test = dataset['test_label'] 
    
    network = TwoLayerNet(784, 50, 10)    

    iters_num = 10000
    train_size = x_train.shape[0]

    batch_size = 100
    learning_rate = 0.1

    optimizer = chainer.optimizers.SGD(lr=learning_rate)
    optimizer.setup(network.net)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        network.gradient(x_batch, t_batch)
        
        optimizer.update()

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc.data, test_acc.data)
      
    print('finish')