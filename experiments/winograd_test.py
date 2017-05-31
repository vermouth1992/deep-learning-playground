"""
test whether winogra algorithm is linear
"""
import tensorflow as tf
import numpy as np


def winograd_2x2_3x3(inputs, filters):
    """

    :param inputs: (4, 4, N)
    :param filter: (3, 3, N)
    :return: 2D conv + sum over
    """
    G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]).astype(np.float32)
    B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).T.astype(np.float32)
    A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).T.astype(np.float32)

    N = inputs.shape[2]

    outputs = np.zeros(shape=(4, 4))

    for i in range(N):
        inputs_transform = B.T.dot(inputs[:, :, i]).dot(B)
        filter_transform = G.dot(filters[:, :, i]).dot(G.T)
        outputs += inputs_transform * filter_transform

    outputs = A.T.dot(outputs).dot(A)

    return outputs


def conv_2x2_3x3_golden(inputs, filters):
    inputs_tf = tf.expand_dims(inputs, axis=0)
    filters_tf = tf.expand_dims(filters, axis=-1)
    outputs = tf.nn.conv2d(inputs_tf, filters_tf, strides=[1, 1, 1, 1], padding='VALID')
    outputs = tf.squeeze(outputs)
    sess = tf.Session()
    return sess.run(outputs)


if __name__ == '__main__':
    inputs = np.random.randn(4, 4, 10).astype(np.float32)
    filters = np.random.randn(3, 3, 10).astype(np.float32)
    outputs_winograd = winograd_2x2_3x3(inputs, filters)
    outputs_golden = conv_2x2_3x3_golden(inputs, filters)
    print outputs_winograd
    print outputs_golden
