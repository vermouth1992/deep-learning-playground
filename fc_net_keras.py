import numpy as np
from collections import Counter

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.engine.topology import Layer


def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim=10, init='he_normal', W_regularizer=l2(0.001)))
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    model = create_model()
    from data_utils import load_CIFAR10, to_categorical

    # logistic regression on cifar10
    cifar10_dir = '/Users/chizhang/Developer/Stanford/tf-playground/dataset/cifar-10-batches-py'
    print 'load cifar10 dataset...'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
    y_train = to_categorical(y_train, 10)
    # X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)
    # y_test = to_categorical(y_test, 10)

    num_training = int(X_train.shape[0] * 0.98)
    train_mask = range(num_training)
    val_mask = range(num_training, X_train.shape[0])

    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-1), metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=1, validation_data=(X_val, y_val))

    score = np.mean((np.argmax(model.predict(X_test), axis=1) == y_test).astype(np.float32))
    print 'Test accuracy', score
