"""
Recurrent Conditional GAN in keras.
It only contains the base class and subclass needs to create their own architecture
"""
from abc import abstractmethod

import tensorflow as tf


class RCGAN(object):
    def __init__(self, input_dim, window_length, num_classes, code_size=64, learning_date=1e-4, batch_size=32,
                 tensorboard=False):
        self.input_dim = input_dim
        self.window_length = window_length
        self.num_classes = num_classes
        self.code_size = code_size
        self.learning_rate = learning_date
        self.batch_size = batch_size
        self.tensorboard = tensorboard

        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()
        self.discriminator_generator = self._combine_generator_discriminator()

    @abstractmethod
    def _create_discriminator(self):
        """ This class also needs to make sure the model is compiled with optimizer """
        raise NotImplementedError('Subclass must implement create discriminator')

    @abstractmethod
    def _create_generator(self):
        """ This class also needs to make sure the model is compiled with optimizer """
        raise NotImplementedError('Subclass must implement create generator')

    @abstractmethod
    def _combine_generator_discriminator(self):
        raise NotImplementedError('Subclass must implement combine generator and discriminator')

    def build_summary(self):
        if self.tensorboard:
            self.dis_loss = tf.Variable(0.)
            tf.summary.scalar('dis_loss', self.dis_loss)
            self.gen_loss = tf.Variable(0.)
            tf.summary.scalar('gen_loss', self.gen_loss)
            self.summary_ops = tf.summary.merge_all()

    @abstractmethod
    def train(self, train_samples, training_labels, num_epoch=5, log_step=50, verbose=True,
              summary_path='./summary/rcgan'):
        pass

    @abstractmethod
    def generate(self, codes, labels):
        pass
