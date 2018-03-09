"""
keras implementation of variational recurrent auto encoder
"""

from keras import backend as K
from keras.layers import Dense, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam


class VariationalRecurrentAutoencoder(object):
    def __init__(self, input_dim, window_length, learning_rate=1e-4, batch_size=128, n_z=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = 100

        self.n_z = n_z
        self.input_dim = input_dim
        self.window_length = window_length
        self.epsilon_std = 1.0

        self.model, self.encoder, self.decoder = self._create_model()

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.n_z), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.sqrt(K.exp(z_log_var)) * epsilon

    def _create_model(self):
        time_series_input = Input(shape=(self.window_length, self.input_dim))
        x = LSTM(512)(time_series_input)
        x = Dense(256, activation='elu')(x)
        z_mu = Dense(self.n_z)(x)

        encoder = Model(inputs=time_series_input, outputs=z_mu)

        z_log_sigma_z_sq = Dense(self.n_z)(x)

        z = Lambda(self._sampling, output_shape=(self.n_z,))([z_mu, z_log_sigma_z_sq])

        decoder_1 = Dense(256, activation='elu')
        repeat_z = RepeatVector(self.window_length)
        decoder_3 = LSTM(512, return_sequences=True)
        decoder_4 = TimeDistributed(Dense(self.input_dim, activation='sigmoid'))

        x = decoder_1(z)
        decode_h = decoder_3(repeat_z(x))
        time_series_hat = decoder_4(decode_h)

        latent_input = Input(shape=(self.n_z,))

        time_series_decode = decoder_4(decoder_3(repeat_z(decoder_1(latent_input))))

        decoder = Model(inputs=latent_input, outputs=time_series_decode)

        model = Model(inputs=time_series_input, outputs=time_series_hat)

        recon_loss = -K.sum(
            time_series_input * K.log(time_series_hat + K.epsilon()) + (1 - time_series_input) * K.log(
                1 - time_series_hat + K.epsilon()), axis=-1)
        recon_loss = K.sum(recon_loss, axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_sigma_z_sq - K.square(z_mu) - K.exp(z_log_sigma_z_sq), axis=-1)
        loss = K.mean(recon_loss + kl_loss)
        model.add_loss(loss)
        model.compile(optimizer=Adam(lr=self.learning_rate))

        return model, encoder, decoder

    def train(self, x_train):
        self.model.fit(x_train, batch_size=self.batch_size, epochs=self.num_epochs, validation_split=0.2, shuffle=True)

    def reconstructor(self, x):
        return self.model.predict(x)

    def generator(self, z):
        return self.decoder.predict(z)

    def transformer(self, x):
        return self.encoder.predict(x)
