"""
keras implementation of Variation auto-encoder for mnist dataset
"""

from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam


class VariationalAutoencoder(object):
    def __init__(self, input_dim, learning_rate=1e-4, batch_size=128, n_z=5):
        # config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = 100

        self.n_z = n_z
        self.input_dim = input_dim
        self.epsilon_std = 1.0

        self.model, self.encoder, self.decoder = self._create_model()

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.n_z), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.sqrt(K.exp(z_log_var)) * epsilon

    def _create_model(self):
        image_input = Input(shape=(self.input_dim,), name='image_input')
        x = Dense(512, activation='elu')(image_input)
        x = Dense(384, activation='elu')(x)
        x = Dense(256, activation='elu')(x)
        z_mu = Dense(self.n_z)(x)
        z_log_sigma_z_sq = Dense(self.n_z)(x)

        z = Lambda(self._sampling, output_shape=(self.n_z,))([z_mu, z_log_sigma_z_sq])

        decoder_1 = Dense(256, activation='elu')
        decoder_2 = Dense(384, activation='elu')
        decoder_3 = Dense(512, activation='elu')
        decoder_4 = Dense(self.input_dim, activation='sigmoid')

        x = decoder_1(z)
        x = decoder_2(x)
        x = decoder_3(x)
        x_hat = decoder_4(x)  # the output should be between 0 and 1

        recon_loss = -K.sum(
            image_input * K.log(x_hat + K.epsilon()) + (1 - image_input) * K.log(1 - x_hat + K.epsilon()), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_sigma_z_sq - K.square(z_mu) - K.exp(z_log_sigma_z_sq), axis=-1)
        loss = K.mean(recon_loss + kl_loss)

        model = Model(inputs=image_input, outputs=x_hat)
        model.add_loss(loss)
        model.compile(optimizer=Adam(lr=self.learning_rate))

        encoder = Model(inputs=image_input, outputs=z_mu)

        latent_input = Input(shape=(self.n_z,))
        x_decode = decoder_4(decoder_3(decoder_2(decoder_1(latent_input))))

        decoder = Model(inputs=latent_input, outputs=x_decode)
        return model, encoder, decoder

    def train(self, x_train):
        self.model.fit(x_train, batch_size=self.batch_size, epochs=self.num_epochs, validation_split=0.2, shuffle=True)

    def reconstructor(self, x):
        return self.model.predict(x)

    def generator(self, z):
        return self.decoder.predict(z)

    def transformer(self, x):
        return self.encoder.predict(x)
