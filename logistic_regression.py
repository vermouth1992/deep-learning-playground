"""
A logistic regression class
"""

from model import MachineLearningModel
import tensorflow as tf
from solver import Solver
import numpy as np


class LogisticRegression(MachineLearningModel):
    def __init__(self, dtype, input_shape, reg):
        super(LogisticRegression, self).__init__(dtype)
        dimension, num_classes = input_shape
        self.X = tf.placeholder(dtype=self.dtype, shape=[None, dimension], name='X')
        self.Y = tf.placeholder(dtype=self.dtype, shape=[None, num_classes], name='Y')
        self.W = tf.Variable(tf.random_normal(input_shape), dtype=self.dtype, name='weights')  # Dx1 matrix
        self.b = tf.Variable(tf.zeros([1, num_classes]), dtype=self.dtype, name='bias')
        self.reg = reg

    def _combine_inputs(self, X):
        return tf.matmul(X, self.W) + self.b

    def inference(self, X):
        return tf.sigmoid(self._combine_inputs(X))

    def loss(self, X, Y):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._combine_inputs(X), labels=Y)) + \
               tf.reduce_sum(tf.multiply(self.W, self.W)) * 0.5

    def evaluate(self, sess, X):
        predicted = tf.argmax(self._combine_inputs(X), axis=1)
        return sess.run(predicted)

    def check_accuracy(self, X, Y):
        predicted = tf.argmax(self._combine_inputs(tf.cast(X, dtype=self.dtype)), axis=1)
        expected = tf.argmax(Y, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(predicted, expected), tf.float32))


if __name__ == '__main__':
    from data_utils import load_CIFAR10, to_categorical

    # logistic regression on cifar10
    cifar10_dir = '/Users/chizhang/Developer/Stanford/tf-stanford-tutorials/playground/dataset/cifar-10-batches-py'
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
        logistic_regress_model = LogisticRegression(dtype=np.float32, input_shape=[3 * 32 * 32, 10], reg=1e-1)
        solver = Solver(logistic_regress_model, training_data, dtype=np.float32, graph=graph, sess=sess,
                        optimizer='adam', optimizer_config={'learning_rate': 1e-3},
                        batch_size=128, num_epochs=100, export_summary=False)
        solver.train()

        w_value, b_value = sess.run([logistic_regress_model.W, logistic_regress_model.b])

        accuracy = sess.run(logistic_regress_model.check_accuracy(X_test, y_test))

        print 'Testing Accuracy: %f' % accuracy
