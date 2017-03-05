"""
A simple convolutional neural network for cifar10
"""
from model import MachineLearningModel
from layers import *
from solver import Solver

class ConvolutionNeuralNet(MachineLearningModel):
    """
    Not meant to be general (a general way is to define a set of layer classes)
    """

    def __init__(self, dtype, input_shape=(32, 32, 3), num_classes=10, weight_scale=1e-2, reg=0.0, use_batch_norm=False):
        super(ConvolutionNeuralNet, self).__init__(dtype)
        self.X = tf.placeholder(dtype=dtype, shape=[None] + list(input_shape))
        self.Y = tf.placeholder(dtype=dtype, shape=[None, num_classes])
        self.reg = reg

        self.weights, self.bias = {}, {}
        #
        self.weights['w1'] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=(3, 3, 3, 32)),
                                          dtype=dtype, name='w1')
        self.bias['b1'] = tf.Variable(initial_value=tf.zeros(shape=[32]), name='b1')
        # self.weights['w2'] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=(3, 3, 32, 32)),
        #                                   dtype=dtype, name='w2')
        # self.bias['b2'] = tf.Variable(initial_value=tf.zeros(shape=[32]), name='b2')
        self.weights['w3'] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=(3, 3, 32, 64)),
                                          dtype=dtype, name='w3')
        self.bias['b3'] = tf.Variable(initial_value=tf.zeros(shape=[64]), name='b3')
        # self.weights['w4'] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=(3, 3, 64, 64)),
        #                                   dtype=dtype, name='w4')
        # self.bias['b4'] = tf.Variable(initial_value=tf.zeros(shape=[64]), name='b4')

        # fully-connected
        self.weights['w5'] = tf.Variable(initial_value=weight_scale * tf.random_normal(shape=[8 * 8 * 64, 512]),
                                          dtype=dtype, name='w5')
        self.bias['b5'] = tf.Variable(initial_value=tf.zeros(shape=[512]), name='b5')
        self.weights['w6'] = tf.Variable(initial_value=weight_scale * tf.truncated_normal(shape=[512, num_classes]))
        self.bias['b6'] = tf.Variable(initial_value=tf.zeros(shape=[num_classes]), name='b6')

        self.is_training = False

    def inference(self, X):
        with tf.name_scope('conv1'):
            out1_1_conv = relu(tf.nn.conv2d(X, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') +
                               self.bias['b1'])
            # output is BATCH_SIZE * 32 * 32 * 32
            # out1_2_conv = relu(tf.nn.conv2d(out1_1_conv, self.weights['w2'], strides=[1, 1, 1, 1], padding='SAME') +
            #                    self.bias['b2'])
            # output is BATCH_SIZE * 32 * 32 * 32
            out1_3_conv = tf.nn.max_pool(out1_1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # output is BATCH_SIZE * 16 * 16 * 32
            # if self.is_training:
            #     out1_3_conv = dropout(out1_3_conv, keep_prob=0.75)

        with tf.name_scope('conv2'):
            out2_1_conv = relu(tf.nn.conv2d(out1_3_conv, self.weights['w3'], strides=[1, 1, 1, 1], padding='SAME') +
                               self.bias['b3'])
            # output is BATCH_SIZE * 16 * 16 * 64
            # out2_2_conv = relu(tf.nn.conv2d(out2_1_conv, self.weights['w4'], strides=[1, 1, 1, 1], padding='SAME') +
            #                    self.bias['b4'])
            # output is BATCH_SIZE * 16 * 16 * 64
            out2_3_conv = tf.nn.max_pool(out2_1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # output is BATCH_SIZE * 8 * 8 * 64
            # if self.is_training:
            #     out2_3_conv = dropout(out2_3_conv, keep_prob=0.75)

        with tf.name_scope('fully_connected'):
            out1_fc = tf.reshape(out2_3_conv, shape=[-1, 8 * 8 * 64])
            out2_fc = relu(affine(out1_fc, self.weights['w5'], self.bias['b5']))
            if self.is_training:
                out2_fc = dropout(out2_fc, keep_prob=0.5)
            out3_fc = affine(out2_fc, self.weights['w6'], self.bias['b6'])

        return out3_fc

    def loss(self, X, Y):
        self.is_training = True
        reg_loss_lst = [tf.nn.l2_loss(self.weights[w]) for w in self.weights]
        reg_loss = self.reg * tf.add_n(reg_loss_lst, 'regularization_loss')
        return reg_loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=self.inference(X)))

    def check_accuracy(self, X, Y):
        self.is_training = False
        predicted = tf.argmax(self.inference(tf.cast(X, dtype=self.dtype)), axis=1)
        expected = tf.argmax(Y, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(predicted, expected), tf.float32))


if __name__ == '__main__':
    from data_utils import load_CIFAR10, to_categorical

    # logistic regression on cifar10
    cifar10_dir = '/Users/chizhang/Developer/Stanford/tf-playground/dataset/cifar-10-batches-py'
    print 'load cifar10 dataset...'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    y_train = to_categorical(y_train, 10)
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
        model = ConvolutionNeuralNet(dtype=tf.float32, input_shape=(32, 32, 3), num_classes=10,
                                  weight_scale=1e-2, reg=1e-7, use_batch_norm=False)
        solver = Solver(model, training_data, dtype=np.float32, graph=graph, sess=sess,
                        optimizer='momentum', optimizer_config={'learning_rate': 1e-1, 'decay_rate': 0.95},
                        batch_size=128, num_epochs=10, export_summary=False)
        solver.train()
        accuracy = sess.run(model.check_accuracy(X_test, y_test))

        print 'Testing Accuracy: %.4f' % accuracy
