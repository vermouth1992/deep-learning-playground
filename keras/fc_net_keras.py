import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam


def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.00)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.00)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(output_dim=100, init='he_normal', W_regularizer=l2(0.00)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(output_dim=10, init='he_normal', W_regularizer=l2(0.00)))
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    model = create_model()
    from keras_utils.data_utils import load_CIFAR10, to_categorical

    # logistic regression on cifar10
    cifar10_dir = '/Users/chizhang/Documents/Deep Learning Resources/datasets/cifar-10-batches-py'
    print 'load cifar10 dataset...'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
    y_train = to_categorical(y_train, 10)
    # X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)
    # y_test = to_categorical(y_test, 10)

    num_training = int(X_train.shape[0] * 0.98)
    train_mask = range(num_training)
    val_mask = range(num_training, X_train.shape[0])
    dev_mask = range(int(X_train.shape[0] * 0.001))  # use to overfit the model

    X_dev = X_train[dev_mask]
    y_dev = y_train[dev_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-1), metrics=['accuracy'])

    model.fit(X_dev, y_dev, batch_size=50, nb_epoch=100, verbose=1, validation_data=(X_val, y_val))

    score = np.mean((np.argmax(model.predict(X_test), axis=1) == y_test).astype(np.float32))
    print 'Test accuracy', score
