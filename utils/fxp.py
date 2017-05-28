import tensorflow as tf
import numpy as np

# Custom truncation layer
def floating_to_fixed_tf(array, bit_length, fraction_len):
    """
    :param array: number or numpy array
    :param integer_length: the signed bit is included
    :param fraction_len:
    :return:
    """
    maximum_value = 2 ** (bit_length - 1) - 1
    minimum_value = -maximum_value - 1
    # normalize array to have no fraction
    array = array * 2 ** fraction_len
    # apply round
    array = tf.round(array)
    # saturation
    array = tf.clip_by_value(array, minimum_value, maximum_value)
    # re-normalize to fixed point
    array = array / 2 ** fraction_len
    return array


# Used for converting weights
def floating_to_fixed_np(array, bit_len, fraction_len):
    """
        :param array: number or numpy array
        :param integer_length: the signed bit is included
        :param fraction_len:
        :return:
        """
    assert isinstance(bit_len, int) and isinstance(fraction_len, int)
    maximum_value = 2 ** (bit_len - 1) - 1
    minimum_value = -maximum_value - 1
    # normalize array to have no fraction
    array *= 2 ** fraction_len
    # apply round
    array = np.round(array)
    # saturation
    array = np.clip(array, minimum_value, maximum_value)
    # re-normalize to fixed point
    array /= 2 ** fraction_len
    return array


def calculate_fxp_quantization(array):
    """

    :param array:
    :return: bit_length, fraction_length
    """
    maximum_value = np.max(np.abs(array))
    if maximum_value < 1:
        return 8, 7
    required_bits = int(np.ceil(np.log2(maximum_value)))
    if required_bits < 7:
        bit_len = 8
    elif required_bits < 16:
        bit_len = 16
    else:
        raise ValueError('Array contains large number which is not suitable for fixed point')
    fraction_len = bit_len - 1 - required_bits
    return bit_len, fraction_len


def truncate_weights(model, verbose=True):
    # truncate weights
    for i in range(len(model.layers)):
        layer = model.get_layer(index=i)
        weights = layer.get_weights()
        if len(weights) != 0:
            if verbose:
                print(layer.name)
            fxp_weights = []
            for weight in weights:
                bit_len, fraction_len = calculate_fxp_quantization(weight)
                if verbose:
                    print(bit_len, fraction_len)
                weight = floating_to_fixed_np(weight, bit_len, fraction_len)
                fxp_weights.append(weight)
            layer.set_weights(fxp_weights)
