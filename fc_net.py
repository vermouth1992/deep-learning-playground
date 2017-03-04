"""
A fully-connected neural net class
"""

from model import MachineLearningModel
import tensorflow as tf
from solver import Solver
import numpy as np
from layers import *


class FullyConnectedNet(MachineLearningModel):
    """
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    """

    def __init__(self, dtype, input_shape, hidden_shape, num_classes, weight_scale=1e-2, reg=0.0, use_batch_norm=False,
                 keep_prob=0.5):
        """

        :param dtype: data type
        :param input_shape: a number specifies the dimension of the input
        :param hidden_shape: a list specifies the dimension of the hidden layer
        :param num_classes:
        :param reg:
        """
        super(FullyConnectedNet, self).__init__(dtype)
        self.X = tf.placeholder(dtype=dtype, shape=[None, input_shape])
        self.Y = tf.placeholder(dtype=dtype, shape=[None, num_classes])
        self.reg = reg
        self.use_batch_norm = use_batch_norm

        if keep_prob > 1.0 or keep_prob <= 0.0:
            self.use_dropout = False
        else:
            self.use_dropout = True
        self.keep_prob = tf.Variable(initial_value=tf.cast(keep_prob, dtype=dtype), dtype=dtype, trainable=False)

        self.is_training = False

        # initialize variable
        weight_shape_lst = [input_shape] + hidden_shape + [num_classes]
        self.num_layers = len(weight_shape_lst) - 1
        self.weights, self.bias = {}, {}
        for i in range(1, len(weight_shape_lst)):
            weight_name = 'w' + str(i)
            weight_shape = [weight_shape_lst[i - 1], weight_shape_lst[i]]
            bias_name = 'b' + str(i)
            bias_shape = [weight_shape_lst[i]]
            self.weights[weight_name] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=weight_shape),
                                                    dtype=self.dtype, name=weight_name)
            self.bias[bias_name] = tf.Variable(initial_value=tf.zeros(shape=bias_shape), dtype=self.dtype,
                                               name=bias_name)

        if self.use_batch_norm:
            self.bn_param = [{} for _ in range(self.num_layers - 1)]

    def inference(self, X):
        output = X
        for i in range(1, self.num_layers):
            weight_name = 'w' + str(i)
            bias_name = 'b' + str(i)
            output = affine(output, self.weights[weight_name], self.bias[bias_name])
            if self.use_batch_norm:
                output = batch_norm(output, is_training=self.is_training, bn_param=self.bn_param[i - 1])
            output = relu(output)
            if self.use_dropout and self.is_training:
                output = dropout(output, self.keep_prob)

        weight_name = 'w' + str(self.num_layers)
        bias_name = 'b' + str(self.num_layers)
        return affine(output, self.weights[weight_name], self.bias[bias_name])

    def loss(self, X, Y):
        self.is_training = True
        reg_loss_lst = [tf.nn.l2_loss(self.weights[w]) for w in self.weights]
        reg_loss = self.reg * tf.add_n(reg_loss_lst, 'regularization_loss')
        return reg_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=self.inference(X)))

    def check_accuracy(self, X, Y):
        self.is_training = True
        predicted = tf.argmax(self.inference(tf.cast(X, dtype=self.dtype)), axis=1)
        expected = tf.argmax(Y, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(predicted, expected), tf.float32))


if __name__ == '__main__':
    from data_utils import load_CIFAR10, to_categorical

    # logistic regression on cifar10
    cifar10_dir = '/Users/chizhang/Developer/Stanford/tf-playground/dataset/cifar-10-batches-py'
    print 'load cifar10 dataset...'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
    y_train = to_categorical(y_train, 10)
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)
    y_test = to_categorical(y_test, 10)

    num_training = int(X_train.shape[0] * 0.98)
    train_mask = range(num_training)
    val_mask = range(num_training, X_train.shape[0])

    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    training_data = {'X_train': X_train, 'y_train': y_train,
                     'X_val': X_val, 'y_val': y_val}

    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        model = FullyConnectedNet(dtype=tf.float32, input_shape=3 * 32 * 32, hidden_shape=[100, 100, 100],
                                  num_classes=10,
                                  weight_scale=1e-2, reg=1e-7, use_batch_norm=True, keep_prob=-1)
        solver = Solver(model, training_data, dtype=np.float32, graph=graph, sess=sess,
                        optimizer='momentum', optimizer_config={'learning_rate': 1e-1, 'decay_rate': 0.95},
                        batch_size=128, num_epochs=10, export_summary=False)
        solver.train()
        accuracy = sess.run(model.check_accuracy(X_test, y_test))

        print 'Testing Accuracy: %.4f' % accuracy
