# encoding: utf-8
# !/usr/bin/python

"""
This file contains utils to create batch file for imagenet 2012 validation data
"""
import os
import shutil
import json
import random
from scipy.misc import imread, imshow

import numpy as np

import h5py

from keras.preprocessing import image

imagenet_val_dir = "~/Documents/Deep Learning Resources/datasets/ILSVRC2012_img_val/"
imagenet_val_label_file = "~/Downloads/imagenet_2012_validation_synset_labels.txt"


def read_class_dictionary():
    f = open(os.path.expanduser('~/.keras/models/imagenet_class_index.json'))
    index_to_class = json.load(f, encoding='utf-8')
    class_index_to_str = {}
    for _, value in index_to_class.iteritems():
        class_index_to_str[value[0]] = value[1]
    return class_index_to_str


def get_class_index_to_index():
    f = open(os.path.expanduser('~/.keras/models/imagenet_class_index.json'))
    index_to_class = json.load(f, encoding='utf-8')
    class_index_to_index_dict = {}
    for key, value in index_to_class.iteritems():
        class_index_to_index_dict[value[0]] = int(key)
    return class_index_to_index_dict


def visualize(image_index, image_classes):
    filename = 'ILSVRC2012_val_' + str(image_index).zfill(8) + '.JPEG'
    filepath = os.path.join(os.path.expanduser(imagenet_val_dir), filename)
    print image_classes
    imshow(imread(filepath))


def move_images():
    imagenet_val_labels = open(os.path.expanduser(imagenet_val_label_file)).read().split()

    for image_index in range(0, 50000):
        # for each image
        filename = 'ILSVRC2012_val_' + str(image_index + 1).zfill(8) + '.JPEG'
        filepath = os.path.join(os.path.expanduser(imagenet_val_dir), filename)
        directory_path = imagenet_val_labels[image_index]
        destination_path = os.path.join(
            os.path.expanduser('~/Documents/Deep Learning Resources/datasets/ILSVRC2012_img_val_224x224/'),
            directory_path, filename)
        shutil.move(filepath, destination_path)


def create_imagenet_batch(image_size, batch_index):
    """
    
    :param image_size: (224, 224) or (299, 299)
    :param batch_index:
    :return: 
    """
    class_index_to_index_dict = get_class_index_to_index()
    class_index_dirs = os.listdir(os.path.expanduser(imagenet_val_dir))

    if image_size[0] == 224:
        batch_size = 5000
    elif image_size[0] == 299:
        batch_size = 2000
    else:
        raise ValueError('Unknown imagesize')

    data = np.empty(shape=[batch_size] + list(image_size) + [3], dtype=np.uint8)
    current_index = 0
    labels = np.empty(shape=(batch_size,), dtype=np.int32)

    num_images_per_batch = batch_size / 1000

    for class_index_dir in class_index_dirs:
        print 'processing', class_index_dir, current_index
        prediction_index = class_index_to_index_dict[class_index_dir]

        image_names = os.listdir(os.path.join(os.path.expanduser(imagenet_val_dir), class_index_dir))
        image_names.sort()
        image_names = image_names[batch_index * num_images_per_batch: (batch_index + 1) * num_images_per_batch]

        for image_name in image_names:
            img_path = os.path.join(os.path.expanduser(imagenet_val_dir), class_index_dir, image_name)
            img = image.load_img(img_path, target_size=image_size)
            x = image.img_to_array(img)
            data[current_index] = x
            labels[current_index] = prediction_index
            current_index += 1
    filename = os.path.expanduser(
        '~/Documents/Deep Learning Resources/datasets/ILSVRC2012_img_val_' + str(image_size[0]) + 'x' +
        str(image_size[1]) + '/val_batch_' + str(batch_index) + '.h5')
    f = h5py.File(filename, 'w')
    f['data'] = data
    f['labels'] = labels
    f.close()

if __name__ == '__main__':
    for batch_index in range(10):
        create_imagenet_batch(image_size=(224, 224), batch_index=batch_index)
