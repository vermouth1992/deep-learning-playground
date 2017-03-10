"""
Transfer learning from tiny-imagenet-100-a to cifar10
"""
import h5py
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.regularizers import l2
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.engine.topology import Layer

from data_utils import *

input_size = (3, 32, 32)
nb_classes = 10

filepath = '/Users/chizhang/Documents/Deep Learning Resources/pretrained model/pretrained_model_tiny_imagenet_100_a.h5'

data = get_CIFAR10_data()

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

y_train = to_categorical(y_train, nb_classes)
y_val = to_categorical(y_val, nb_classes)


def tiny_imagenet_100_a_model():
    # read h5py
    with h5py.File(filepath) as f:
        conv_weight_lst = []
        conv_bias_lst = []
        conv_gamma_lst = []
        conv_beta_lst = []
        conv_running_mean_lst = []
        conv_running_var_lst = []
        for i in range(1, 11):
            weight_name = 'W%d' % (i)
            bias_name = 'b%d' % (i)
            gamma_name = 'gamma%d' % (i)
            beta_name = 'beta%d' % (i)
            running_mean_name = 'running_mean%d' % (i)
            running_var_name = 'running_var%d' % (i)
            conv_weight_lst.append(f[weight_name][:])
            conv_bias_lst.append(f[bias_name][:])
            conv_gamma_lst.append(f[gamma_name][:])
            conv_beta_lst.append(f[beta_name][:])
            conv_running_mean_lst.append(f[running_mean_name][:])
            conv_running_var_lst.append(f[running_var_name][:])

        model = Sequential()

        pad_lst = [2, 1, 1, 1, 1, 1, 1, 1, 1]

        for i in range(0, 9):
            nb_filter, num_channel, filter_height, filter_width = conv_weight_lst[i].shape
            gamma, beta, running_mean, running_var = conv_gamma_lst[i], conv_beta_lst[i], conv_running_mean_lst[i], \
                                                     conv_running_var_lst[i]
            if i % 2 == 0:
                subsample = (2, 2)
            else:
                subsample = (1, 1)

            pad = pad_lst[i]
            if i == 0:
                model.add(ZeroPadding2D(padding=(pad, pad), input_shape=input_size, trainable=False))
            else:
                model.add(ZeroPadding2D(padding=(pad, pad), trainable=False))

            conv_layer = Convolution2D(nb_filter, filter_height, filter_width, border_mode='valid',
                                       name='conv%d' % (i + 1), subsample=subsample,
                                       weights=[conv_weight_lst[i], conv_bias_lst[i]], trainable=False)
            model.add(conv_layer)
            model.add(BatchNormalization(weights=[gamma, beta, running_mean, running_var], axis=1, trainable=False))
            model.add(Activation('relu', trainable=False))

        # fc layers
        model.add(Flatten())

        return model


def classifier(l2_reg, keep_prob=0.5):
    model = Sequential()
    model.add(Dense(512, init='he_normal', input_shape=(1024,), W_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(1.0 - keep_prob))

    model.add(Dense(128, init='he_normal', W_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(1.0 - keep_prob))

    model.add(Dense(nb_classes, init='he_normal', W_regularizer=l2(l2_reg)))
    model.add(Activation('softmax'))

    return model


def extract_features():
    """ extract features for cifar10 """
    model = tiny_imagenet_100_a_model()

    training_features = model.predict(X_train, verbose=1)
    validation_features = model.predict(X_val, verbose=1)
    testing_features = model.predict(X_test, verbose=1)

    with h5py.File('cifar10_features.h5', 'w') as f:
        f.create_dataset(name='training_features', data=training_features)
        f.create_dataset(name='validation_features', data=validation_features)
        f.create_dataset(name='testing_features', data=testing_features)



if __name__ == '__main__':

    f = h5py.File('cifar10_features.h5', 'r')
    X_train_features = f['training_features'][:]
    X_validation_features = f['validation_features'][:]
    X_test_features = f['testing_features'][:]
    f.close()

    model = classifier(l2_reg=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-2, decay=1e-5), metrics=['accuracy'])
    model.fit(X_train_features, y_train, batch_size=256, nb_epoch=20, verbose=1,
              validation_data=(X_validation_features, y_val))

    score = np.mean((model.predict_classes(X_test_features) == y_test).astype(np.float32))
    print 'Test accuracy', score
