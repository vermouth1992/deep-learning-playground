"""
Use a 3 layer perceptron to simulate a max function over a vector
"""

import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


nb_classes = 16

def create_dataset(num_samples, num_features):
    data = np.random.random_sample((num_samples, num_features)) * 2 - 1
    labels = np.argmax(data, axis=1)
    num_training = int(num_samples * 0.8)
    return (data[0:num_training], labels[0:num_training]), (data[num_training:], labels[num_training:])


def create_network():
    model = Sequential()
    model.add(Dense(512, input_shape=(nb_classes,)))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (training_data, training_labels), (testing_data, testing_labels) = create_dataset(50000, nb_classes)
    training_labels = np_utils.to_categorical(training_labels, nb_classes)
    testing_labels = np_utils.to_categorical(testing_labels, nb_classes)
    model = create_network()
    model.fit(training_data, training_labels, batch_size=128, nb_epoch=20,
              validation_data=(testing_data, testing_labels), shuffle=True)
    model.evaluate()