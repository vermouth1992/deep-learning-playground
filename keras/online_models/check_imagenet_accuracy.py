import sys
sys.path.append('../../')

from utils.fxp import truncate_weights, floating_to_fixed_np
from utils.imagenet_utils import get_imagenet_batch

import numpy as np
from keras import metrics

from inception_v3 import InceptionV3
from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19
from xception import Xception

if __name__ == '__main__':

    model_name = 'vgg16'

    if model_name == 'vgg16':
        model = VGG16
    elif model_name == 'vgg19':
        model = VGG19
    elif model_name == 'resnet50':
        model = ResNet50
    elif model_name == 'inception_v3':
        model = InceptionV3
    elif model_name == 'xception':
        model = Xception
    else:
        raise ValueError('Unknown CNN model')

    if model_name in ['vgg16', 'vgg19', 'resnet50']:
        imagesize = 224
        total_num_batch = 10
    elif model_name in ['inception_v3', 'xception']:
        imagesize = 299
        total_num_batch = 25
    else:
        raise ValueError('Unknown CNN model')

    model = model(include_top=True, weights='imagenet', truncate=True)

    truncate_weights(model)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])

    accuracy_lst, top_5_accuracy_lst = [], []

    for i in range(total_num_batch):
        x, y = get_imagenet_batch(i, imagesize)

        x = floating_to_fixed_np(x, bit_len=16, fraction_len=7)

        print('Batch index:', i)

        _, accuracy, top_5_accuracy = model.evaluate(x, y, batch_size=64)
        print('Accuracy', accuracy)
        print('Top 5 accuracy', top_5_accuracy)
        accuracy_lst.append(accuracy)
        top_5_accuracy_lst.append(top_5_accuracy)

        del x
        del y

    print('Total accuracy:', np.mean(accuracy_lst))
    print('Total top 5 accuracy:', np.mean(top_5_accuracy_lst))
