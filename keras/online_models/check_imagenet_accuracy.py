import os
from keras.utils import np_utils
from keras import metrics

from vgg16 import VGG16
from vgg19 import VGG19
from resnet50 import ResNet50
from inception_v3 import InceptionV3
from xception import Xception

import numpy as np
import h5py

model_name = 'inception_v3'

def preprocess_input_inception(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

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
    from keras.applications.imagenet_utils import preprocess_input
    total_num_batch = 10
elif model_name in ['inception_v3', 'xception']:
    imagesize = 299
    preprocess_input = preprocess_input_inception
    total_num_batch = 25
else:
    raise ValueError('Unknown CNN model')


model = model(include_top=True, weights='imagenet')

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', metrics.top_k_categorical_accuracy])

accuracy_lst, top_5_accuracy_lst = [], []

def get_batch(index):
    filepath = os.path.expanduser('~/Documents/Deep Learning Resources/datasets/imagenet/ILSVRC2012_img_val_' + str(imagesize) +
                                  'x' + str(imagesize) + '/val_batch_' + str(index) + '.h5')
    f = h5py.File(filepath, 'r')
    x = f['data'][:].astype('float32')
    y = f['labels'][:].astype('int32')
    y = np_utils.to_categorical(y, num_classes=1000)
    x = preprocess_input(x)
    return x, y

for i in range(total_num_batch):
    x, y = get_batch(i)
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
