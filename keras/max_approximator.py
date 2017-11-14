"""
Use a 3 layer perceptron to simulate a max function over a vector
"""

import numpy as np
from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils


nb_classes = 4

def normalize(data):
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    return (data - np.expand_dims(min_data, axis=1)) / np.expand_dims(max_data - min_data, axis=1) * 2 - 1

def create_dataset(num_samples, num_features):
    data = np.random.random_sample((num_samples, num_features)) * 2 - 1
    labels = np.argmax(data, axis=1)
    num_training = int(num_samples * 0.8)
    return (data[0:num_training], labels[0:num_training]), (data[num_training:], labels[num_training:])


def create_network():
    model = Sequential()
    model.add(Dense(128, input_shape=(nb_classes,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (training_data, training_labels), (testing_data, testing_labels) = create_dataset(1500, nb_classes)
    training_labels = np_utils.to_categorical(training_labels, nb_classes)
    testing_labels = np_utils.to_categorical(testing_labels, nb_classes)
    model = create_network()
    model.fit(training_data, training_labels, batch_size=128, epochs=100,
              validation_data=(testing_data, testing_labels), shuffle=True)

    data = np.random.randn(4000, nb_classes)
    labels = np_utils.to_categorical(np.argmax(data, axis=1), nb_classes)
    acc = model.evaluate(data, labels)