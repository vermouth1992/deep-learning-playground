
import tensorflow as tf

class MachineLearningModel(object):
    """
    An abstract class defines API for subclasses and solver
    """
    def __init__(self, dtype, input_shape):
        """

        :param dtype: data type used to initialize placeholder
        :param kwargs: how to init the trainable variables, varies in different models
        """
        self.dtype = dtype
        if type(input_shape) is not list:
            raise TypeError('input_shape must be a list')
        dimension, num_classes = input_shape
        self.X = tf.placeholder(dtype=self.dtype, shape=[None, dimension], name='X')
        self.Y = tf.placeholder(dtype=self.dtype, shape=[None, num_classes], name='Y')

    def inference(self, X):
        """
        forward pass, X is (N * d1 * d2 * ... * dn), can be reshape according to requirement
        :return: scores (a value in computation graph)
        """
        raise NotImplementedError('Must be implemented by subclasses')

    def loss(self, X, Y):
        """
        compute the loss given training data, use function defined above to implement
        :return: loss (a value in computation graph)
        """
        raise NotImplementedError('Must be implemented by subclasses')

    def evaluate(self, sess, X):
        """
        evaluate the performance (classification or regression) given input X in sess
        :param sess: session to evaluate
        :param X: input data
        :return: scores
        """
        raise NotImplementedError('Must be implemented by subclasses')

    def check_accuracy(self, X, Y):
        """
        check the accuracy given X (data) and Y (label)
        :param sess: session to evaluate
        :param X: input data
        :param Y: input label
        :return: accuracy
        """
        raise NotImplementedError('Must be implemented by subclasses')