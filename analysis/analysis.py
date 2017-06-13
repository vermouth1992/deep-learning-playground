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


def analyze_convolutional_layer(layer):
    assert isinstance(layer, Conv2D), 'Must be Conv2D layer'



def analyze_fc_layer(layer):
    pass


def analyze_batch_normalization_layer(layer):
    pass


def analyze_pooling_layer(layer):
    pass


def analyze_dropout_layer(layer):
    pass


def analyze_no_effect_layer(layer):
    pass


analyze_routine_dict = {Conv2D: analyze_convolutional_layer,
                        Dense: analyze_fc_layer,
                        BatchNormalization: analyze_batch_normalization_layer,
                        MaxPooling2D: analyze_pooling_layer,
                        Dropout: analyze_dropout_layer}


class LayerStatistics:
    def __init__(self, layer):
        assert isinstance(layer, Layer), "The layer must be a keras layer"
        self.layer = layer
        self.analysis_result = None

    def _analyze(self):
        analyze_routine = analyze_routine_dict.get(type(self.layer), None)
        if analyze_routine is not None:
            self.analysis_result = analyze_routine(self.layer)


class ModelStatistics:
    def __init__(self):
        pass
