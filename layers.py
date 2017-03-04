"""
This file defines some wrappers for deep neural net layers
"""

import tensorflow as tf
import numpy as np


def affine(X, W, b):
    return tf.matmul(X, W) + b


def relu(X):
    return tf.nn.relu(X)


def batch_norm(X, is_training, momentum=0.9, epsilon=1e-5):
    scale = tf.Variable(tf.ones([X.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([X.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([X.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([X.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(X, [0])
        train_mean = tf.assign(pop_mean, tf.multiply(pop_mean, momentum) + batch_mean * (1 - momentum))
        train_var = tf.assign(pop_var, tf.multiply(pop_var, momentum) + batch_var * (1 - momentum))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(X, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(X, pop_mean, pop_var, beta, scale, epsilon)


def dropout(X, keep_prob):
    return tf.nn.dropout(X, keep_prob=keep_prob)
