"""
Deep Convolutional Generative Adversarial Network using Keras
"""
from __future__ import print_function

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, InputLayer, Flatten, Dense, Reshape, \
    Activation
from keras.models import Sequential
from keras.optimizers import RMSprop


class DCGAN(object):
    def __init__(self):
        # config
        self.code_size = 64
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epoch = 5
        self.log_step = 50

        self.discriminator = self._create_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))
        self.generator = self._create_generator()
        self.discriminator_generator = self._combine_generator_discriminator()
        self.discriminator_generator.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))

    def _create_discriminator(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(32, 32, 3)))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))  # the probability that it is real or not
        return model

    def _create_generator(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(64,)))
        model.add(Dense(4 * 4 * 128))
        model.add(Reshape(target_shape=(4, 4, 128)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('sigmoid'))
        return model

    def _combine_generator_discriminator(self):
        model = Sequential()
        model.add(self.generator)
        self.discriminator.trainable = False
        model.add(self.discriminator)
        return model

    def train(self, train_samples):
        num_train = train_samples.shape[0]
        step = 0

        # smooth the loss curve so that it does not fluctuate too much
        smooth_factor = 0.95
        plot_dis_s = 0
        plot_gen_s = 0
        plot_ws = 0

        dis_losses = []
        gen_losses = []
        for epoch in range(self.num_epoch):
            for i in range(num_train // self.batch_size):
                step += 1

                self.discriminator.trainable = True

                batch_samples = train_samples[i * self.batch_size: (i + 1) * self.batch_size]
                noise = np.random.normal(0, 1, [self.batch_size, self.code_size])
                generated_images = self.generator.predict(noise, verbose=0)
                X = np.concatenate((batch_samples, generated_images))
                zeros = np.zeros([self.batch_size])
                ones = np.ones([self.batch_size])
                y = np.concatenate((ones, zeros))
                dis_loss = self.discriminator.train_on_batch(X, y)

                self.discriminator.trainable = False

                gen_loss = self.discriminator_generator.train_on_batch(noise, ones)

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                dis_losses.append(plot_dis_s / plot_ws)
                gen_losses.append(plot_gen_s / plot_ws)

                if step % self.log_step == 0:
                    print('Iteration {0}: dis loss = {1:.4f}, gen loss = {2:.4f}'.format(step, dis_loss, gen_loss))
        return dis_losses, gen_losses

    def generate_one_sample(self, code):
        return self.generator.predict(code, verbose=0)

    def generate(self, codes):
        generated = np.zeros((codes.shape[0], 32, 32, 3))
        for i in range(codes.shape[0]):
            generated[i:i + 1] = self.generate_one_sample(codes[i:i + 1])
        return generated


if __name__ == '__main__':
    pass
