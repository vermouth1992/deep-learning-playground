"""
Train a affine layer for extracted features by vgg16
"""

from keras.models import Model, Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input

import numpy as np

from utils.data_utils import *
from vgg16 import VGG16


def extract_features(layer):
    base_model = VGG16(include_top=False, weights='imagenet', classes=10)
    model = Model(base_model.input, outputs=base_model.get_layer(layer).output)
    print("Loading cifar10 dataset")
    cifar10_dataset = get_CIFAR10_data(subtract_mean=False)
    X_train, y_train = cifar10_dataset['X_train'], cifar10_dataset['y_train']
    X_val, y_val = cifar10_dataset['X_val'], cifar10_dataset['y_val']
    X_test, y_test = cifar10_dataset['X_test'], cifar10_dataset['y_test']

    X_train = preprocess_input(X_train).transpose(0, 2, 3, 1)
    X_val = preprocess_input(X_val).transpose(0, 2, 3, 1)
    X_test = preprocess_input(X_test).transpose(0, 2, 3, 1)

    X_train_feature = model.predict(X_train, batch_size=128, verbose=1)
    X_val_feature = model.predict(X_val, batch_size=128, verbose=1)
    X_test_feature = model.predict(X_test, batch_size=128, verbose=1)

    np.savez('cifar10_feature_' + layer, X_train=X_train_feature, X_val=X_val_feature, X_test=X_test_feature,
             y_train=y_train, y_val=y_val, y_test=y_test)


def affine_model():
    input_shape = (4, 4, 256)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, name='fc1', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(10, name='prediction', activation='softmax', kernel_initializer='he_normal'))
    return model


if __name__ == '__main__':

    do_extract_features = False
    do_to_categorical = True
    layer = 'block3_pool'
    filename = 'cifar10_vgg16_weights.h5'

    if do_extract_features:
        extract_features(layer)
    else:
        # load data
        dataset = np.load('cifar10_feature_' + layer + '.npz')
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        if do_to_categorical:
            y_train = np_utils.to_categorical(y_train, 10)
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        if do_to_categorical:
            y_val = np_utils.to_categorical(y_val, 10)
        X_test = dataset['X_test']
        y_test = dataset['y_test']

        # create model
        model = affine_model()

        try:
            model.load_weights(filename)
            predict = model.predict_classes(X_test, batch_size=128)
            print("Accuracy = " + str(np.mean(predict == y_test)))
        except:
            pass

        optimizer = Adam(lr=5e-5, decay=1e-7)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # train
        y_test = np_utils.to_categorical(y_test)

        while True:
            num_epoch = raw_input("Type in number of epoch\n")
            model.fit(X_train, y_train, batch_size=128, epochs=int(num_epoch), validation_data=(X_test, y_test),
                      shuffle=True)
            text = raw_input("Continue to train?\n")
            if text != 'yes' and text != 'y':
                break

        # check accuracy
        # predict = model.predict_classes(X_test, batch_size=128)
        # print("Accuracy = " + str(np.mean(predict == y_test)))
        text = raw_input("Save weights?\n")
        if text == 'yes' or text == 'y':
            print('Saving weights...')
            model.save_weights(filename)
