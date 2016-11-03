'''

File containing implementation of layers not implemented in tensorflow

'''

import tensorflow as tf

def space_to_batch_1d(input, paddings, block_size, name=None):
    '''
    1d equivalent of space_to_batch function in tensorflow
    Args:
        input: tensor of shape [batch_size, in_width, in_depth]
        paddings: a list of left and right zero padding for the in_width dimension in the input tensor
        block_size: sub-sampling rate in the in_width dimension of the data
        name: function name scope for tensorflow

    Returns:
    Tensor of shape [bath_size*block_size,in_width_pad/block_size,in_depth]
    '''

    with tf.name_scope(name, "space_to_batch_1d") as name:

        if block_size < 1:
            raise ValueError("block_size must be greater than 1")

        ip_shape = tf.shape(input)
        padded = tf.pad(input,paddings)
        reshaped = tf.reshape(padded,[-1,block_size,ip_shape[2]])
        permuted = tf.transpose(reshaped,[1,0,2])
        return tf.reshape(permuted,[ip_shape[0]*block_size,-1,ip_shape[2]])

def batch_to_space_1d(input, crop, block_size, name=None):

    with tf.name_scope(name, "batch_to_space_1d") as name:

        if block_size < 1:
            raise ValueError("block_size must be greater than 1")

        ip_shape = tf.shape(input)
        batch_size = ip_shape[0]/block_size
        reshaped = tf.reshape(input,[block_size,-1,ip_shape[2]])
        permuted = tf.transpose(reshaped,[1,0,2])
        reshaped2 = tf.reshape(permuted, [batch_size,-1,ip_shape[2]])
        return tf.crop(reshaped2,crop)

def artrous_conv1d(value, filters, rate, padding, name=None):
    '''
    1d equivalent of the tensorflow artrous 2d convolution

    Args:
        value: input 3-D tensor with shape [batch_size, in_width, ip_channels]
        filters: convolution filter with shape [filter_width, ip_channels, op_channels]
        rate: rate of dilation, 1 is equivalent to normal convolution
        padding: string, "SAME" or "VALID" only
        name: function scoping name for tensorflow

    Returns:
    Tensor of shape [batch_size, new_width, op_channels]
    '''
    with tf.name_scope(name, "artrous_conv1d") as name:
        if rate == 1:
            # perform normal convolution
            return tf.nn.conv1d(value,filters,stride=1,padding=padding)

        elif rate > 1:
            # determine padding based on padding choice


            # reshape value tensor based on type of padding and rate

            # perform 1d convolution

            # re-shape the output to output size


        else:
            raise ValueError("Rate must be >= 1")