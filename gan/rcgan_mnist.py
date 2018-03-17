"""
RCGAN architecture for mnist. We view mnist as 28 timestamp, each with dimension 28
"""

import numpy as np
from keras import layers
from keras.layers import Input, Embedding, Flatten, Dense, RepeatVector, LSTM, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam

from rcgan import RCGAN


class RCGANMNIST(RCGAN):
    def _create_discriminator(self):
        time_series_input = Input(shape=(self.window_length, self.input_dim),
                                  name='time_series_input')  # the input is 28 timestamp, each 28 dimension
        x = LSTM(64)(time_series_input)
        x = Dense(32, activation='elu')(x)
        fake = Dense(1, activation='sigmoid', name='generation')(x)
        aux = Dense(self.num_classes, activation='softmax', name='auxiliary')(x)
        model = Model(inputs=time_series_input, outputs=[fake, aux])
        model.compile(optimizer=Adam(lr=self.learning_rate, beta_1=0.5),
                                   loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
        return model

    def _create_generator(self):
        latent_input = Input(shape=(self.code_size,))
        image_class = Input(shape=(1,), dtype='int32')
        # train an embedding layer
        cls = Flatten()(
            Embedding(self.num_classes, self.code_size, embeddings_initializer='glorot_normal')(image_class))
        h = layers.multiply([latent_input, cls])  # shape is (None, code_size)
        # build a sequential
        decoder_1 = Dense(32, activation='elu')
        repeat_z = RepeatVector(self.window_length)
        decoder_3 = LSTM(64, return_sequences=True)
        decoder_4 = TimeDistributed(Dense(self.input_dim, activation='tanh'))

        time_series_decode = decoder_4(decoder_3(repeat_z(decoder_1(h))))

        model = Model(inputs=[latent_input, image_class], outputs=time_series_decode)
        model.compile(loss='binary_crossentropy', optimizer=Adam(self.learning_rate, beta_1=0.5))
        return model

    def _combine_generator_discriminator(self):
        latent = Input(shape=(self.code_size,))
        image_class = Input(shape=(1,), dtype='int32')
        fake = self.generator([latent, image_class])
        self.discriminator.trainable = False
        fake, aux = self.discriminator(fake)
        combined = Model([latent, image_class], [fake, aux])
        combined.compile(
            optimizer=Adam(lr=self.learning_rate, beta_1=0.5),
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )
        return combined

    def train(self, train_samples, training_labels, num_epoch=5, log_step=100, verbose=True,
              summary_path='./summary/rcgan'):
        num_train = train_samples.shape[0]
        step = 0

        # smooth the loss curve so that it does not fluctuate too much
        smooth_factor = 0.95
        plot_dis_s = 0
        plot_gen_s = 0
        plot_ws = 0

        dis_losses = []
        gen_losses = []

        for epoch in range(num_epoch):
            disc_sample_weight = [np.ones(2 * self.batch_size),
                                  np.concatenate((np.ones(self.batch_size) * 2, np.zeros(self.batch_size)))]
            for i in range(num_train // self.batch_size):
                step += 1
                # get image
                image_batch = train_samples[i * self.batch_size: (i + 1) * self.batch_size]
                # get label
                label_batch = training_labels[i * self.batch_size: (i + 1) * self.batch_size]
                # get noise
                noise = np.random.normal(-1, 1, [self.batch_size, self.code_size])
                # get sample labels
                sampled_labels = np.random.randint(0, self.num_classes, self.batch_size)
                # get a batch of fake images
                generated_images = self.generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
                x = np.concatenate((image_batch, generated_images))
                soft_zero, soft_one = 0, 0.95
                y = np.array([soft_one] * self.batch_size + [soft_zero] * self.batch_size)
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
                dis_loss = self.discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight)[0]
                # print(dis_loss)
                noise = np.random.normal(-1, 1, (2 * self.batch_size, self.code_size))
                sampled_labels = np.random.randint(0, self.num_classes, 2 * self.batch_size)
                trick = np.ones(2 * self.batch_size) * soft_one
                gen_loss = self.discriminator_generator.train_on_batch([noise, sampled_labels.reshape((-1, 1))],
                                                                       [trick, sampled_labels])[0]
                # print(gen_loss)
                plot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                dis_losses.append(plot_dis_s / plot_ws)
                gen_losses.append(plot_gen_s / plot_ws)

                if step % log_step == 0 and verbose:
                    print('Iteration {0}: dis loss = {1:.4f}, gen loss = {2:.4f}'.format(step, dis_loss, gen_loss))
        return dis_losses, gen_losses

    def generate(self, codes, labels):
        return self.generator.predict([codes, labels])

    def save_model(self, path='./weights/rcgan/'):
        self.generator.save_weights(path + '/generator_mnist.h5')
        self.discriminator.save_weights(path + '/discriminator_mnist.h5')

    def load_model(self, path='./weights/rcgan/'):
        self.generator.load_weights(path + '/generator_mnist.h5')
        self.discriminator.load_weights(path + '/discriminator_mnist.h5')