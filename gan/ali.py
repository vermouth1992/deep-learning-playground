"""
Adversarial Learned inference in Keras
"""

from __future__ import print_function

import numpy as np
import keras
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, InputLayer, Flatten, Dense, Reshape, \
    Activation, Input, Embedding
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

class ALI(object):
    def __init__(self, code_size, num_channels):
        self.code_size = code_size
        self.num_channels = num_channels

    def _create_inference(self):
        image_input = Input(shape=(32, 32, self.num_channels), name='image_input')
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
        label_embedding = Flatten()(self.label_embedding(label_input))

        x = keras.layers.concatenate([x, label_embedding])

        z_avg = Dense(self.code_size)(x)
        z_log_var = Dense(self.code_size)(x)
        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)

        model = Model([image_input, label_input], [z_avg, z_log_var])

        return model

    def _create_generator(self):
        # combine at the input directly
        noise_input = Input(shape=(self.code_size,), name='noise_input')
        label_input = Input(shape=(10,), name='label_input')
        self.label_embedding = Embedding(input_dim=10, output_dim=self.code_size)
        label_embedding = Flatten()(self.label_embedding(label_input))
        x = keras.layers.concatenate([noise_input, label_embedding])

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
        x = Conv2DTranspose(filters=self.num_channels, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer=RandomNormal(0, 0.02))(x)
        output = Activation('sigmoid')(x)
        model = Model(inputs=[noise_input, label_input], outputs=output)
        return model

    def _create_discriminator(self):
        pass
