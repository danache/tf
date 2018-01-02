import tensorflow as tf

import numpy as np
def _conv_block(inputs, numOut, name='conv_block',is_training=True):
    """ Convolutional Block
    Args:
        inputs	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the block
    Returns:
        conv_3	: Output Tensor
    """

    with tf.name_scope(name):
        with tf.name_scope('norm_1'):
            norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                  is_training=is_training)
            conv_1 = _conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
        with tf.name_scope('norm_2'):
            norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                  is_training=is_training)
            pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
            conv_2 = _conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
        with tf.name_scope('norm_3'):
            norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                  is_training=is_training)
            conv_3 = _conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
        return conv_3


def _skip_layer( inputs, numOut, name='skip_layer'):
    """ Skip Layer
    Args:
        inputs	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the bloc
    Returns:
        Tensor of shape (None, inputs.height, inputs.width, numOut)
    """
    with tf.name_scope(name):
        if inputs.get_shape().as_list()[3] == numOut:
            return inputs
        else:
            conv = _conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
            return conv

def Residual( inputs, numOut, name = 'residual_block',is_training=True):

    with tf.name_scope(name):
        convb = _conv_block(inputs, numOut)
        skipl = _skip_layer(inputs, numOut)

        return tf.add_n([convb, skipl], name='res_block')


def _conv(inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
    """ Spatial Convolution (CONV2D)
    Args:
        inputs			: Input Tensor (Data Type : NHWC)
        filters		: Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name			: Name of the block
    Returns:
        conv			: Output Tensor (Convolved Input)
    """
    with tf.name_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
            [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        return conv

