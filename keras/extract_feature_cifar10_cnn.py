"""
This file is aimed to extract output of each layer
"""
from keras.models import Model

import h5py

from cnn_cifar10 import create_model, get_cifar10_dataset


if __name__ == '__main__':
    f = h5py.File('cifar-10_output.h5', 'w')
    x_test_group = f.create_group("x_test_group")

    base_model = create_model()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_cifar10_dataset()

    x_train = None

    raw_input('Press any key to continue...\n')

    for i in range(0, len(base_model.layers)):
        layer_name = base_model.get_layer(index=i).name
        print layer_name
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(index=i).output)
        x_test_feature = model.predict(x_test, batch_size=64, verbose=1)
        x_test_group.create_dataset(name=layer_name, data=x_test_feature)

    f.close()
