#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib.request
import pickle
import os
import gzip
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
dataset_dir = './mnist'

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result.astype(np.float32)

def load_img(file_name):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)

    return data

def load_label(file_name):
    file_path = dataset_dir + '/' + file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels

def show_image(dataset):
    for i in range(dataset.size):
        example = dataset[i].reshape((28, 28))
        plt.imshow(example, 'gray')
        plt.pause(0.01)

def to_onehot(dataset):
    ret = np.zeros((dataset.size, 10), dtype=np.int32)
    for i in range(dataset.size):
        ans = dataset[i]
        ret[i][ans] = 1
    return ret

def load_dataset():
    save_file = dataset_dir + '/mnist.pkl'
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        for v in key_file.values():
            file_path = dataset_dir + '/' + v
            urllib.request.urlretrieve(url_base + v, file_path)
        
        dataset = {}
        dataset['train_img'] = min_max(load_img(key_file['train_img']), axis=1)
        dataset['train_label'] = load_label(key_file['train_label']).astype(np.int32)
        dataset['test_img'] = min_max(load_img(key_file['test_img']), axis=1)
        dataset['test_label'] = load_label(key_file['test_label']).astype(np.int32)
        
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)

    return dataset