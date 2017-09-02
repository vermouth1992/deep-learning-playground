"""
A set of utility functions to get the actual computation complexity of various layers
"""

import math
import warnings

oneGiga = 1e9


def convolutional_layer_direct(imageSize, filterSize, padding='SAME', stride=(1, 1), inGFLOP=True,
                               result_format='mac'):
    """ The input conforms to tensorflow tf.nn.conv2d format

    :param imageSize: (image_height, image_width, in_channels), represented as a tuple
    :param filterSize: (filter_height, filter_width, in_channels, out_channels)
    :param padding:
    :param stride: (height_stride, width_stride)
    :return: The number of operations: accumulation + multiplication
    """
    image_height, image_width, in_channels = imageSize
    assert in_channels == filterSize[2], "image size and filter size must have the same depth."
    filter_height, filter_width, _, out_channels = filterSize
    height_stride, width_stride = stride
    if padding == 'SAME':
        height_padding = (image_height - 1) * height_stride - image_height + filter_height
        width_padding = (image_width - 1) * width_stride - image_width + filter_width
    elif padding == 'VALID':
        height_padding = 0
        width_padding = 0
    else:
        raise ValueError('Unknown padding')

    out_height = (image_height - filter_height + height_padding) // height_stride + 1
    out_width = (image_width - filter_width + width_padding) // width_stride + 1
    # number of operations to get one result
    numOpMac = filter_height * filter_width * in_channels
    numOpPerResult = 2 * numOpMac + in_channels + 1  # 1 is the bias
    total_num_result = out_height * out_width * out_channels
    if result_format == 'mac':
        total_num_ops = total_num_result * numOpMac
    elif result_format == 'op':
        total_num_ops = total_num_result * numOpPerResult
    else:
        raise ValueError('Unknown result format')
    if inGFLOP:
        total_num_ops /= oneGiga
    return total_num_ops


def isPowerOfTwo(num):
    return ((num & (num - 1)) == 0) and num > 0


def findUnitSize(size):
    """
    >>> findUnitSize(3)
    2
    >>> findUnitSize(7)
    2
    """
    assert size >= 3, "It is not valid to use fft when size < 3"
    F = size
    start = int(math.ceil(math.log(F, 2)))
    while True:
        total = 2 ** start
        unitSize = total + 1 - F
        if unitSize >= 2:
            break
        start += 1
    return unitSize


def convolutional_layer_fft(imageSize, filterSize, padding='SAME', stride=(1, 1), inGFLOP=True,
                            result_format='mac', fft_size=None):
    """ The input conforms to tensorflow tf.nn.conv2d format

    :param imageSize: (image_height, image_width, in_channels), represented as a tuple
    :param filterSize: (filter_height, filter_width, in_channels, out_channels)
    :param padding:
    :param stride: (height_stride, width_stride)
    :param fft_size: find the minimum feasible fft size if it is None, (height_fft_size, width_fft_size)
    :return: The number of operations: accumulation + multiplication
    """
    image_height, image_width, in_channels = imageSize
    filter_height, filter_width, _, out_channels = filterSize
    assert in_channels == filterSize[2], "image size and filter size must have the same depth."
    if fft_size is None:
        height_unit_size = findUnitSize(filter_height)
        width_unit_size = findUnitSize(filter_width)
        height_fft_size = height_unit_size + filter_height - 1
        width_fft_size = width_unit_size + filter_width - 1
    else:
        height_fft_size, width_fft_size = fft_size
        height_unit_size = height_fft_size + 1 - filter_height
        width_unit_size = width_fft_size + 1 - filter_width

    # number of operations
    height_logFFTUnitSize = int(math.log(height_fft_size, 2))
    width_logFFTUnitSize = int(math.log(width_fft_size, 2))

    height_stride, width_stride = stride

    if padding == 'SAME':
        height_padding = (image_height - 1) * height_stride - image_height + filter_height
        width_padding = (image_width - 1) * width_stride - image_width + filter_width
    elif padding == 'VALID':
        height_padding = 0
        width_padding = 0
    else:
        raise ValueError('Unknown padding')

    numTilt = int(math.ceil((image_height + height_padding) / float(height_unit_size))) * \
              int(math.ceil((image_width + width_padding) / float(width_unit_size)))
    # Note that this is complex multiplication, 1 complex multiplication = 3 real multiplication
    num_multiply_per_complex_multiply = 3
    num_add_per_complex_add = 2

    numMultImageFFT = (height_fft_size * width_fft_size * height_logFFTUnitSize +
                       width_fft_size * height_fft_size * width_logFFTUnitSize) * numTilt * in_channels * \
                      num_multiply_per_complex_multiply / 2.0
    numMultIFFT = (height_fft_size * width_fft_size * height_logFFTUnitSize +
                   width_fft_size * height_fft_size * width_logFFTUnitSize) * numTilt * out_channels * \
                  num_multiply_per_complex_multiply / 2.0
    # numAddImageFFT = (height_fft_size * height_logFFTUnitSize * width_fft_size +
    #                   width_fft_size * width_logFFTUnitSize * height_fft_size) * numTilt * in_channels * \
    #                  num_add_per_complex_add
    numMultImageFilter = height_fft_size * width_fft_size * in_channels * out_channels * numTilt * \
                         num_multiply_per_complex_multiply
    numAddInDepth = numMultImageFilter
    numAddOverlap = ((filter_height - 1) * width_fft_size + (filter_width - 1) * height_fft_size) * \
                    numTilt * out_channels * 2  # multiply by 2 because each tile overlapped by 4 boundary

    if inGFLOP:
        numMultImageFFT /= oneGiga
        numMultImageFilter /= oneGiga
        numMultIFFT /= oneGiga
        numAddInDepth /= oneGiga
        numAddOverlap /= oneGiga

    if result_format == 'mac':
        return numMultImageFFT, numMultImageFilter, numMultIFFT, numAddOverlap
    elif result_format == 'op':
        return numMultImageFFT, numMultImageFilter + numAddInDepth, numMultIFFT, numAddOverlap
    else:
        raise ValueError('Unknown result format')


def convolutional_layer_winograd(imageSize, filterSize, padding='SAME', stride=(1, 1), inGFLOP=True,
                                 result_format='mac'):
    """ The input conforms to tensorflow tf.nn.conv2d format

    :param imageSize: (image_height, image_width, in_channels), represented as a tuple
    :param filterSize: (filter_height, filter_width, in_channels, out_channels)
    :param padding:
    :param stride: (height_stride, width_stride)
    :param fft_size: find the minimum feasible fft size if it is None, (height_fft_size, width_fft_size)
    :return: The number of operations: accumulation + multiplication
    """
    image_height, image_width, in_channels = imageSize
    filter_height, filter_width, _, out_channels = filterSize
    assert in_channels == filterSize[2], "image size and filter size must have the same depth."
    assert filter_height == 3 and filter_width == 3, 'Winograd is only applicable to 3x3 filters'
    if stride != (1, 1):
        warnings.warn('For stride not equal to 1, it is generally not recommended to use Winograd algorithm')
    height_stride, width_stride = stride
    if padding == 'SAME':
        height_padding = (image_height - 1) * height_stride - image_height + filter_height
        width_padding = (image_width - 1) * width_stride - image_width + filter_width
    elif padding == 'VALID':
        height_padding = 0
        width_padding = 0
    else:
        raise ValueError('Unknown padding')

    num_tile = ((image_height + height_padding) / 2 - 1) * ((image_width + width_padding) / 2 - 1)
    # for each 4x4 transform, B'dB
    num_accumulation_per_transform_result = 2   # based on the paper, intermediate result can be reused
    num_transform_result = 16
    transform_add_per_tile = num_accumulation_per_transform_result * num_transform_result
    total_transform_add = transform_add_per_tile * num_tile * in_channels
    # mac
    total_num_mac = num_transform_result * in_channels * num_tile * out_channels
    # inverse transform
    num_accumulation_per_inverse_transform_result = 6 # based on the paper, intermediate result can be reused
    num_inverse_transform_result = 4
    total_inverse_transform_add = num_accumulation_per_inverse_transform_result * \
                                  num_inverse_transform_result * num_tile * out_channels

    if inGFLOP:
        total_transform_add /= oneGiga
        total_num_mac /= oneGiga
        total_inverse_transform_add /= oneGiga

    if result_format == 'mac':
        return total_transform_add, total_num_mac, total_inverse_transform_add
    elif result_format == 'op':
        return total_transform_add, total_num_mac * 2, total_inverse_transform_add
    else:
        raise ValueError('Unknown result format')


def fc_layer(input_width, output_width, inGFLOP=True, result_format='mac'):
    total_num_mac = input_width * output_width
    if inGFLOP:
        total_num_mac /= oneGiga
    if result_format == 'mac':
        return total_num_mac
    elif result_format == 'op':
        return total_num_mac * 2
    else:
        raise ValueError('Unknown result format')


def batch_normalization_layer(input_shape, inGLOP=True):
    N, D = input_shape
    total_num_add = 2 * N * D
    total_num_multiplication = 2 * N * D
    if inGLOP:
        total_num_add /= oneGiga
        total_num_multiplication /= oneGiga
    return total_num_add, total_num_multiplication
