'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
import keras.backend as K

from keras_utils.data_utils import get_CIFAR10_data

batch_size = 128
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

reg = 0

# The data, shuffled and split between train and test sets:
data = get_CIFAR10_data()
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)


def create_model_1():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, 32, 32), kernel_regularizer=l2(reg),
                            name='conv1'))
    model.add(BatchNormalization(axis=1, name='conv1_batch'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(reg), name='fc1'))
    model.add(BatchNormalization(name='fc1_batch'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, kernel_regularizer=l2(reg), name='fc2'))
    model.add(Activation('softmax'))

    return model


def create_model_2():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(3, 32, 32), kernel_regularizer=l2(reg),
                            name='conv1'))
    model.add(BatchNormalization(axis=1, name='conv1_batch'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(reg), name='conv2'))
    model.add(BatchNormalization(axis=1, name='conv2_batch'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(reg), name='fc1'))
    model.add(BatchNormalization(name='fc1_batch'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, kernel_regularizer=l2(reg), name='fc2'))
    model.add(Activation('softmax'))

    return model


def train_model():
    model_index = 1

    model = create_model_1()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    weight_file_name = 'cifar10_cnn_weights_%d.h5' % (model_index)

    try:
        model.load_weights(weight_file_name)
    except:
        pass

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_val, Y_val),
              shuffle=True)

    model.save_weights(weight_file_name)

    predicted_labels = model.predict_classes(X_test, verbose=0)
    accuracy = np.mean((predicted_labels == y_test).astype(np.float32))
    print 'Test accuracy:', accuracy


def model_ensemble():

    def check_accuracy(score, y):
        return np.mean((np.argmax(score, axis=1) == y).astype(np.float32))

    model_1 = create_model_1()
    model_1.load_weights('cifar10_cnn_weights_1.h5')
    model_2 = create_model_2()
    model_2.load_weights('cifar10_cnn_weights_2.h5')

    print '========Ensemble validation data========'
    score_1 = model_1.predict(X_val, verbose=0)
    acc_1 = check_accuracy(score_1, y_val)
    score_2 = model_2.predict(X_val, verbose=0)
    acc_2 = check_accuracy(score_2, y_val)
    avg_score = np.add(score_1, score_2) / 2
    avg_acc = check_accuracy(avg_score, y_val)
    print 'model 1 acc', acc_1
    print 'model 2 acc', acc_2
    print 'ensemble acc', avg_acc
    print '========Ensemble test data========'
    score_1 = model_1.predict(X_test, verbose=0)
    acc_1 = check_accuracy(score_1, y_test)
    score_2 = model_2.predict(X_test, verbose=0)
    acc_2 = check_accuracy(score_2, y_test)
    avg_score = np.add(score_1, score_2) / 2
    avg_acc = check_accuracy(avg_score, y_test)
    print 'model 1 acc', acc_1
    print 'model 2 acc', acc_2
    print 'ensemble acc', avg_acc


if __name__ == '__main__':
    model_ensemble()
