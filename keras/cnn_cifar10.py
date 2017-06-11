'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import sys

import keras
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('../')
from utils.fxp import floating_to_fixed_tf, truncate_weights
from utils.data_utils import get_CIFAR10_data

batch_size = 32
num_classes = 10
data_augmentation = True
weight_filepath = "cifar-10.h5"


def get_cifar10_dataset():
    # The data, shuffled and split between train and test sets:
    data = get_CIFAR10_data(cifar10_dir='~/Documents/Deep Learning Resources/datasets/cifar-10-batches-py',
                            num_training=50000, num_validation=0, num_test=10000)
    x_train = data['X_train'].transpose(0, 2, 3, 1)
    y_train = data['y_train']
    x_val = data['X_val'].transpose(0, 2, 3, 1)
    y_val = data['y_val']
    x_test = data['X_test'].transpose(0, 2, 3, 1)
    y_test = data['y_test']

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_val /= 255
    x_test /= 255

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_model():
    model = Sequential()

    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 6), input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
                     input_shape=(32, 32, 3)))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 5)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 2)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 2)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 2)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    # model.add(Lambda(lambda x: floating_to_fixed_tf(x, 8, 1)))
    model.add(Activation('softmax'))

    try:
        model.load_weights(weight_filepath, by_name=True)
    except:
        pass

    # initiate RMSprop optimizer
    opt = keras.optimizers.adam(lr=5e-4, decay=1e-6)

    # Let's train the model using Adam
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = create_model()
    # truncate_weights(model)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_cifar10_dataset()
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=10,
                  validation_data=(x_val, y_val),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        is_train = input("Is train or not?\n")

        if is_train:

            while True:
                num_epoch = raw_input("Type in number of epoch\n")
                # Fit the model on the batches generated by datagen.flow().
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=int(num_epoch),
                                    validation_data=(x_test, y_test))

                text = raw_input("Save weights?\n")
                if text == 'yes' or text == 'y':
                    print('Saving weights...')
                    model.save_weights(weight_filepath)

                text = raw_input("Continue to train?\n")
                if text != 'yes' and text != 'y':
                    break

        else:
            # check accuracy
            print('Checking accuracy...')
            loss, accuracy = model.evaluate(x_test, y_test)
            print("Accuracy = " + str(accuracy))
