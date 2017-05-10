"""
This file defines some wrappers for deep neural net layers
"""

import tensorflow as tf
import numpy as np


def affine(X, W, b):
    return tf.matmul(X, W) + b


def relu(X):
    return tf.nn.relu(X)


def batch_norm(X, is_training, bn_param, momentum=0.9, epsilon=1e-5):
    if 'scale' in bn_param:
        scale = bn_param['scale']
    else:
        scale = tf.Variable(tf.ones([X.get_shape()[-1]]))
        bn_param['scale'] = scale
    if 'beta' in bn_param:
        beta = bn_param['beta']
    else:
        beta = tf.Variable(tf.zeros([X.get_shape()[-1]]))
        bn_param['beta'] = beta
    if 'pop_mean' in bn_param:
        pop_mean = bn_param['pop_mean']
    else:
        pop_mean = tf.Variable(tf.zeros([X.get_shape()[-1]]), trainable=False)
        bn_param['pop_mean'] = pop_mean
    if 'pop_var' in bn_param:
        pop_var = bn_param['pop_var']
    else:
        pop_var = tf.Variable(tf.zeros([X.get_shape()[-1]]), trainable=False)
        bn_param['pop_var'] = pop_var

    if is_training:
        batch_mean, batch_var = tf.nn.moments(X, [0])
        train_mean = tf.assign(pop_mean, tf.multiply(pop_mean, momentum) + batch_mean * (1 - momentum))
        train_var = tf.assign(pop_var, tf.multiply(pop_var, momentum) + batch_var * (1 - momentum))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(X, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(X, pop_mean, pop_var, beta, scale, epsilon)


def spatial_batch_norm(X, is_training, bn_param, momentum=0.9, epsilon=1e-5):
    num_channel = X.get_shape().as_list()[-1]
    out = tf.reshape(X, [-1, num_channel])
    out = batch_norm(out, is_training, bn_param, momentum, epsilon)
    return tf.reshape(out, shape=tf.shape(X))


def dropout(X, keep_prob):
    return tf.nn.dropout(X, keep_prob=keep_prob)

