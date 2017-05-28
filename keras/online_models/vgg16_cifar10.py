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
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.utils import np_utils

import numpy as np

from keras_utils.data_utils import *
from vgg16 import VGG16
from vgg19 import VGG19
from inception_v3 import InceptionV3


def preprocess_input_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def extract_features(base_model, layer, feature_filename, model_name):
    model = Model(base_model.input, outputs=base_model.get_layer(layer).output)
    print("Loading cifar10 dataset")
    cifar10_dataset = get_CIFAR10_data(cifar10_dir='/home/chi/Documents/Deep Learning Resources/datasets/cifar-10-batches-py', subtract_mean=False)
    X_train, y_train = cifar10_dataset['X_train'], cifar10_dataset['y_train']
    X_val, y_val = cifar10_dataset['X_val'], cifar10_dataset['y_val']
    X_test, y_test = cifar10_dataset['X_test'], cifar10_dataset['y_test']

    if model_name == 'inception_v3' or model_name == 'xception':
        preprocess_input = preprocess_input_inception
    else:
        from keras.applications.imagenet_utils import preprocess_input

    X_train = preprocess_input(X_train).transpose(0, 2, 3, 1)
    X_val = preprocess_input(X_val).transpose(0, 2, 3, 1)
    X_test = preprocess_input(X_test).transpose(0, 2, 3, 1)

    X_train_feature = model.predict(X_train, batch_size=128, verbose=1)
    X_val_feature = model.predict(X_val, batch_size=128, verbose=1)
    X_test_feature = model.predict(X_test, batch_size=128, verbose=1)

    np.savez(feature_filename, X_train=X_train_feature, X_val=X_val_feature, X_test=X_test_feature,
             y_train=y_train, y_val=y_val, y_test=y_test)


def affine_model():
    input_shape = (2, 2, 512)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(2048, name='fc1', kernel_initializer='he_normal', kernel_regularizer=l2(0)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(2048, name='fc2', kernel_initializer='he_normal', kernel_regularizer=l2(0)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(10, name='prediction', activation='softmax', kernel_initializer='he_normal'))
    return model


if __name__ == '__main__':

    do_extract_features = False
    do_to_categorical = True
    model_name = 'vgg19'
    layer = 'block4_pool'
    filename = 'cifar10_' + model_name + '_weights_' + layer + '.h5'
    feature_filename = 'cifar10_feature_' + model_name + '_' + layer + '.npz'

    if do_extract_features:
        base_model = VGG19(include_top=False, weights='imagenet', classes=10)
        extract_features(base_model, layer, feature_filename, model_name)
    else:
        # load data
        dataset = np.load(feature_filename)
        X_train = dataset['X_train'][40000:]
        y_train = dataset['y_train'][40000:]
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

        optimizer = Adam(lr=1e-5, decay=1e-9)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        while True:
            num_epoch = raw_input("Type in number of epoch\n")
            model.fit(X_train, y_train, batch_size=128, epochs=int(num_epoch), validation_data=(X_val, y_val),
                      shuffle=True)
            text = raw_input("Continue to train?\n")
            if text != 'yes' and text != 'y':
                break

        # check accuracy
        predict = model.predict_classes(X_test, batch_size=128)
        print("Accuracy = " + str(np.mean(predict == y_test)))
        text = raw_input("Save weights?\n")
        if text == 'yes' or text == 'y':
            print('Saving weights...')
            model.save_weights(filename)
