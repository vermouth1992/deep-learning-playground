"""
This file contains a set of analysis class and helper functions to analyze a Keras model including
- The computation cost of each layer when using FFT approach, Winograd approach and direct convolution
  It should be break down as Transform + MAC + Inverse Transform
- The number of parameters in each layer (original form + transformed form)
- The expected processing time based on the dataflow model of hardware
- The optimal quantization scheme of each layer
"""

from keras.engine.topology import Layer
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout

from utils.computation_complexity import *


class LayerStatistics(object):
    """ A class contains the statistics of a layer """
    def __init__(self):
        pass




def analyze_convolutional_layer(layer):
    assert isinstance(layer, Conv2D), 'Must be Conv2D layer'
    input_shape = layer.input_shape
    config = layer.get_config()
    kernel_size = config['kernel_size']
    in_channels = input_shape[-1]
    out_channels = config['filters']

    imageSize = input_shape[1:]
    filterSize = (kernel_size[0], kernel_size[1], in_channels, out_channels)
    padding = config['padding'].capitalize()
    stride = config['strides']

    direct_conv_mac = convolutional_layer_direct(imageSize, filterSize, padding, stride)
    numMultImageFFT, numMultImageFilter, numMultIFFT, numAddOverlap = convolutional_layer_fft(imageSize,
                                                                                              filterSize,
                                                                                              padding,
                                                                                              stride)
    total_transform_add, total_num_mac, total_inverse_transform_add = convolutional_layer_winograd(imageSize,
                                                                                                   filterSize,
                                                                                                   padding,
                                                                                                   stride)

def analyze_fc_layer(layer):
    pass


def analyze_batch_normalization_layer(layer):
    pass


def analyze_no_computation_layer(layer):
    pass


analyze_routine_dict = {Conv2D: analyze_convolutional_layer,
                        Dense: analyze_fc_layer,
                        BatchNormalization: analyze_batch_normalization_layer,
                        MaxPooling2D: analyze_no_computation_layer,
                        Dropout: analyze_no_computation_layer}

#
# class LayerStatisticsManager:
#     def __init__(self, layer):
#         assert isinstance(layer, Layer), "The layer must be a keras layer"
#         self.layer = layer
#         self.analysis_result = None
#
#     def _analyze(self):
#         analyze_routine = analyze_routine_dict.get(type(self.layer), None)
#         if analyze_routine is not None:
#             self.analysis_result = analyze_routine(self.layer)


class ModelStatistics:
    def __init__(self):
        pass
