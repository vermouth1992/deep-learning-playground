# /usr/bin/python

import h5py
import os

weight_path = os.path.expanduser('~/Developer/deep-learning-playground/keras/cifar-10.h5')

def get_weights():
    """
    :return: a dictionary contains weights of each part
    """
    f = h5py.File(weight_path, 'r')
    result = {}
    for i in range(4):
        conv_name = 'conv2d_%d' % (i + 1)
        result[conv_name] = {}
        result[conv_name]['kernel'] = f[conv_name][conv_name]['kernel:0'][:]
        result[conv_name]['bias'] = f[conv_name][conv_name]['bias:0'][:]
    for i in range(2):
        dense_name = 'dense_%d' % (i + 1)
        result[dense_name] = {}
        result[dense_name]['kernel'] = f[dense_name][dense_name]['kernel:0'][:]
        result[dense_name]['bias'] = f[dense_name][dense_name]['bias:0'][:]
    for i in range(4):
        batch_norm_name = 'batch_normalization_%d' % (i + 1)
        batch_norm_group = f[batch_norm_name][batch_norm_name]
        result[batch_norm_name] = {}
        result[batch_norm_name]['beta'] = batch_norm_group['beta:0'][:]
        result[batch_norm_name]['gamma'] = batch_norm_group['gamma:0'][:]
        result[batch_norm_name]['moving_mean'] = batch_norm_group['moving_mean:0'][:]
        result[batch_norm_name]['moving_variance'] = batch_norm_group['moving_variance:0'][:]
    return result

if __name__ == '__main__':
    result = get_weights()
