# coding: utf-8
import urllib.request
import gzip
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
dataset_dir = './mnist'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def download_mnist():
    for v in key_file.values():
        file_path = dataset_dir + '/' + v
        urllib.request.urlretrieve(url_base + v, file_path)

def load_img(file_name):
    file_path = dataset_dir + '/' + key_file[file_name]
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 784)

    return data

def load_label(file_name):
    file_path = dataset_dir + '/' + key_file[file_name]
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels