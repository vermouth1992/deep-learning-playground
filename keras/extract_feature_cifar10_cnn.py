"""
This file is aimed to extract output of each layer
"""
import numpy as np
from keras.models import Model
from cnn_cifar10 import create_model, get_cifar10_dataset


if __name__ == '__main__':

    base_model = create_model()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_cifar10_dataset()

    x_train = None

    raw_input('Press any key to continue...\n')

    activations_range = {}
    for layer in base_model.layers:
        print(layer.name)
        if 'activation' in layer.name:
            model = Model(inputs=base_model.input, outputs=layer.output)
            activations = model.predict(x_test, verbose=1)
            print('maximum = ', np.max(np.abs(activations)))
            activations_range[layer.name] = np.max(np.abs(activations))

            raw_input('Press any key to continue')

            del activations
            del model

