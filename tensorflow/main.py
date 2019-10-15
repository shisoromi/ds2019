# coding: utf-8
import mnist
import pickle
import os
import matplotlib.pyplot as plt
import tf_train


def load_dataset():
    save_file = './mnist/mnist.pickle'
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            print('load pickle')
            dataset = pickle.load(f)
            print('done')
    else:
        print('cannot find mnist sample data. downloading...')
        mnist.download_mnist()
        print('done')
        print('load image data')
        dataset = {}
        dataset['train_img'] = mnist.load_img('train_img')
        dataset['train_label'] = mnist.load_label('train_label')
        dataset['test_img'] = mnist.load_img('test_img')
        dataset['test_label'] = mnist.load_label('test_label')
        print('done')
        with open(save_file, 'wb') as f:
            print('save to pickle')
            pickle.dump(dataset, f, -1)
            print('done')

    return dataset


if __name__ == "__main__":

    dataset = load_dataset()

    tr = tf_train.Trainer()
    for epoch, loss, accuracy in tr.train(dataset, 10000):
        print("epoch : %d / loss : %f / accuracy : %f" %
              (epoch, loss, accuracy))
