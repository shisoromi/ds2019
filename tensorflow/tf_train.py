# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result.astype(np.float32)


def to_one_hot_label(x):
    ret = np.zeros((x.size, 10), dtype=np.float32)
    for idx in range(x.size):
        target = x[idx]
        ret[idx][target] = 1
    return ret


class Trainer:
    def __init__(self):
        with tf.device("/CPU:0"):
            self.w = {}
            self.b = {}
            self.w['w1'] = tf.Variable(tf.random_normal(
                [784, 50], mean=0.0, stddev=0.05, dtype=tf.float32))
            self.b['b1'] = tf.Variable(tf.zeros([1, 50], dtype=tf.float32))

            self.w['w2'] = tf.Variable(tf.random_normal(
                [50, 10], mean=0.0, stddev=0.05, dtype=tf.float32))
            self.b['b2'] = tf.Variable(tf.zeros([1, 10], dtype=tf.float32))

    def train(self, dataset, iters_num):
        train_img = dataset['train_img']
        train_label = dataset['train_label']
        test_img = dataset['test_img']
        test_label = dataset['test_label']

        train_img_norm = min_max(train_img, 1)
        one_hot_label = to_one_hot_label(train_label)

        test_img_norm = min_max(test_img, 1)
        test_one_hot_label = to_one_hot_label(test_label)

        saver = tf.train.Saver(
            [self.w['w1'], self.w['w2'], self.b['b1'], self.b['b2']])

        with tf.device("/CPU:0"), tf.name_scope('summary'), tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            input_data = tf.placeholder("float32", [None, 784], name='input')
            target_data = tf.placeholder("float32", [None, 10], name='target')
            z1 = tf.nn.relu(tf.matmul(input_data, self.w['w1']) + self.b['b1'])
            z2 = tf.nn.softmax(tf.matmul(z1, self.w['w2']) + self.b['b2'])

            error = tf.reduce_sum((z2 - target_data)**2,
                                  axis=1, name='square_sum_error')
            # sqrtを使うとback propergationでnanになることがあるので二乗和のままにする
            # sqrt_error = tf.sqrt(error, name='sqrt_sum_error')
            # loss = tf.reduce_mean(sqrt_error, axis=0, name='sqrt_ave_error')
            loss = tf.reduce_mean(error, axis=0, name='sqrt_ave_error')

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train_output = optimizer.minimize(loss)

            loss_summary = tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./logs', sess.graph)

            ckpt = tf.train.get_checkpoint_state('./model/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print("load " + last_model)
                saver.restore(sess, last_model)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            train_size = train_img.shape[0]
            batch_size = 100
            iter_per_epoch = max(train_size/batch_size, 1)

            for i in range(iters_num):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = train_img_norm[batch_mask]
                t_batch = one_hot_label[batch_mask]

                _, summary, curr_loss = sess.run([train_output, merged, loss], feed_dict={
                                                 input_data: x_batch, target_data: t_batch})
                if np.isnan(curr_loss):
                    break

                writer.add_summary(summary, i)
                saver.save(sess, './model/model', global_step=100)
                if i % iter_per_epoch == 0:
                    _z2, curr_loss = sess.run([z2, loss], feed_dict={
                        input_data: test_img_norm, target_data: test_one_hot_label})

                    predicted = np.argmax(_z2, axis=1)
                    num_of_ans = np.sum(predicted == test_label)
                    accuracy = num_of_ans / test_label.shape[0]

                    yield i, curr_loss, accuracy

            writer.flush()
            writer.close()
