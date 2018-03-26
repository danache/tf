# coding:utf8
import numpy as np
import tensorflow as tf


class hourglassnet(object):
    def __init__(self, stacks=2, is_training=True, num_keypoints=14):
        self.img_shape = 256
        self.is_training = is_training
        # self.act_fn = tf.nn.relu
        self.block_bn_act_fn = None
        self.stacks = stacks
        self.num_keypoints = num_keypoints

    def hourglass(self, inputs, channels, depth, name='hourglass', reuse=None):
        with tf.variable_scope(name_or_scope=name) as scope:
            id = 0
            out = self._hg_scope(inputs, channels, depth, id, parent_scope=scope, reuse=reuse)
            return out

    def _hg_scope(self, inputs, channels, depth, id, parent_scope, reuse):
        with tf.variable_scope('block_{}'.format(id)) as scope:
            # Upper Branch
            up_1 = self._bottleneck(inputs, channels, name='up_1', reuse=reuse)
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._bottleneck(low_, channels, name='low_1', reuse=reuse)

            if depth > 1:
                low_2 = self._hg_scope(low_1, channels, depth - 1, id + 1, scope, reuse=reuse)
            else:
                low_2 = self._bottleneck(low_1, channels, name='low_2', reuse=reuse)

            low_3 = self._bottleneck(low_2, channels, name='low_3', reuse=reuse)
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')

            return tf.add_n([up_2, up_1], name='out_hg')

    def _conv_block(self, inputs, channels, name='conv_block', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope('conv_bn_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=self.block_bn_act_fn,
                                                      is_training=self.is_training)
                conv_1 = self._conv(norm_1, channels, kernel_size=1, strides=1, pad='VALID', name='conv')
            with tf.variable_scope('conv_bn_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=self.block_bn_act_fn,
                                                      is_training=self.is_training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, channels, kernel_size=3, strides=1, pad='VALID', name='conv')
            with tf.variable_scope('conv_bn_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=self.block_bn_act_fn,
                                                      is_training=self.is_training)
                conv_3 = self._conv(norm_3, channels * 2, kernel_size=1, strides=1, pad='VALID', name='conv')
            return conv_3

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # Kernel for convolution, Xavier Initialisation
            # print("---conv---")
            # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
            #     [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            kernel = tf.get_variable('weights',
                                     shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     trainable=True)
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return conv

    def _skip_layer(self, inputs, channels, name='skip_layer', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            if inputs.get_shape().as_list()[3] == channels * 2:
                return inputs
            else:
                conv = self._conv(inputs, channels * 2, kernel_size=1, strides=1, name='skip_conv')
                return conv

    def _bottleneck(self, inputs, channels, name='bottleneck', reuse=False):
        with tf.variable_scope(name_or_scope=name):
            convb = self._conv_block(inputs, channels, reuse=reuse)
            # print('convb:', convb)
            skipl = self._skip_layer(inputs, channels, reuse=reuse)
            # print('skipl:', skipl)
            return tf.add_n([convb, skipl], name='bottleneck_out')

    def build(self, inputs,reuse=False):
        # inputs = tf.placeholder(tf.float32, [8, 64, 64, 256], name='inputs')
        #
        # outs = self.hourglass(inputs, channels=128, depth=4)
        #inputs = tf.placeholder(tf.float32, [8, 256, 256, 3], name='inputs')
        outs = self.net(inputs,reuse=reuse)
        return outs
    '''
    生成网络
    '''

    def net(self, inputs, reuse=None):
        with tf.variable_scope(name_or_scope='net', reuse=reuse):
            with tf.variable_scope(name_or_scope='preprocessing', reuse=reuse):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128',
                                           reuse=reuse)
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._bottleneck(conv1, channels=64, name='r1', reuse=reuse)
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = self._bottleneck(pool1, channels=64, name='r2', reuse=reuse)
                r3 = self._bottleneck(r2, channels=128, name='r3', reuse=reuse)
            out = []
            with tf.variable_scope(name_or_scope='stacks', reuse=reuse):
                for i in range(self.stacks):
                    # i = 0
                    y = self.hourglass(r3, channels=128, depth=4, name='hg{}'.format(i), reuse=reuse)
                    y = self._make_residual(y, channels=128, num_blocks=4, name='res{}'.format(i), reuse=reuse)
                    y = self._conv_bn_relu(y, 256, 1, 1, pad='SAME', name='fc{}'.format(i), reuse=reuse)
                    score = self._conv_with_bias(y, self.num_keypoints, name='score{}'.format(i), reuse=reuse)
                    out.append(score)
                    if i < self.stacks - 1:
                        fc_ = self._conv_with_bias(y, 256, name='fc_{}'.format(i), reuse=reuse)
                        score_ = self._conv_with_bias(score, 256, name='score_{}'.format(i), reuse=reuse)
                        r3 = tf.add_n([r3, fc_, score_], name='out_stacks')
                return out

    def _make_residual(self, inputs, channels, num_blocks, name='res', reuse=None):
        with tf.variable_scope(name_or_scope=name):
            _inputs = inputs
            for i in range(num_blocks):
                _inputs = self._bottleneck(_inputs, channels, name='res_bot{}'.format(i), reuse=reuse)
            return _inputs

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                    [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
                conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
                norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                    is_training=self.is_training)
                return norm

    def _conv_with_bias(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='conv_with_bias', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # Kernel for convolution, Xavier Initialisation
            # print("---conv---")
            # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
            #     [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            kernel = tf.get_variable('weights',
                                     shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     trainable=True)
            bias = tf.get_variable('biases', [filters], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            out = tf.nn.bias_add(conv, bias)
            return out
