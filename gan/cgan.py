"""
Condition GAN based on DCGAN architecture
"""

from __future__ import print_function

import keras
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape, \
    Activation, Input
from keras.models import Model
from keras.optimizers import RMSprop


class CGAN(object):
    def __init__(self):
        # config
        self.code_size = 64
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epoch = 5
        self.log_step = 50

        self.generator = self._create_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))
        self.discriminator = self._create_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))

        self.discriminator.trainable = False
        self.discriminator_generator = self._combine_generator_discriminator()
        self.discriminator_generator.compile(loss='binary_crossentropy', optimizer=RMSprop(self.learning_rate))

    def _create_discriminator(self):
        image_input = Input(shape=(32, 32, 3), name='image_input')
        x = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=RandomNormal(0, 0.02))(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=RandomNormal(0, 0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                   kernel_initializer=RandomNormal(0, 0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(128, kernel_initializer=RandomNormal(0, 0.02))(x)

        label_input = Input(shape=(10,), name='label_input')
        label_dense = Dense(128, kernel_initializer=RandomNormal(0, 0.02))(label_input)

        x = keras.layers.concatenate([x, label_dense])

        x = Dense(1, kernel_initializer=RandomNormal(0, 0.02))(x)

        output = Activation('sigmoid')(x)  # the probability that it is real or not

        model = Model(inputs=[image_input, label_input], outputs=output)

        return model

    def _create_generator(self):
        # combine at the input directly
        noise_input = Input(shape=(self.code_size,), name='noise_input')
        label_input = Input(shape=(10,), name='label_input')
        x = keras.layers.concatenate([noise_input, label_input])

        x = Dense(4 * 4 * 128, kernel_initializer=RandomNormal(0, 0.02))(x)
        x = Reshape(target_shape=(4, 4, 128))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=RandomNormal(0, 0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=RandomNormal(0, 0.02))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=RandomNormal(0, 0.02))(x)
        output = Activation('sigmoid')(x)
        model = Model(inputs=[noise_input, label_input], outputs=output)
        return model

    def _combine_generator_discriminator(self):
        noise_input = Input(shape=(self.code_size,))
        label_input = Input(shape=(10,), name='label_input')
        image_input = self.generator(inputs=[noise_input, label_input])
        output = self.discriminator(inputs=[image_input, label_input])
        model = Model(inputs=[noise_input, label_input], outputs=output)
        return model

    def train(self, train_samples, training_labels):
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
                # get image
                batch_samples = train_samples[i * self.batch_size: (i + 1) * self.batch_size]
                # get label
                label_samples = training_labels[i * self.batch_size: (i + 1) * self.batch_size]

                noise = np.random.normal(0, 1, [self.batch_size, self.code_size])
                generated_images = self.generator.predict([noise, label_samples], verbose=0)

                self.discriminator.trainable = True

                dis_loss_images = self.discriminator.train_on_batch([batch_samples, label_samples],
                                                                    [1] * self.batch_size)
                dis_loss_noise = self.discriminator.train_on_batch([generated_images, label_samples],
                                                                   [0] * self.batch_size)
                dis_loss = dis_loss_images + dis_loss_noise

                self.discriminator.trainable = False

                sampled_labels = np.random.randint(low=0, high=10, size=self.batch_size)
                sampled_labels = keras.utils.to_categorical(sampled_labels, 10)

                gen_loss = self.discriminator_generator.train_on_batch([noise, sampled_labels], [1] * self.batch_size)

                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                dis_losses.append(plot_dis_s / plot_ws)
                gen_losses.append(plot_gen_s / plot_ws)

                if step % self.log_step == 0:
                    print('Iteration {0}: dis loss = {1:.4f}, gen loss = {2:.4f}'.format(step, dis_loss, gen_loss))
        return dis_losses, gen_losses

    def generate_one_sample(self, code, label):
        return self.generator.predict([code, label], verbose=0)

    def generate(self, codes, labels):
        generated = np.zeros((codes.shape[0], 32, 32, 3))
        for i in range(codes.shape[0]):
            generated[i:i + 1] = self.generate_one_sample(codes[i:i + 1], labels[i:i + 1])
        return generated
