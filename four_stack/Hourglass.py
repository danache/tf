
import tensorflow as tf
import numpy as np



class HourglassModel():
    def __init__(self, nFeats=256, nStack=4, nModules=1, outputDim=14,nLow=4,training=True,CELOSS=False,
                 ):

        self.nStack = nStack
        self.nFeats = nFeats
        self.nModules = nModules
        self.partnum = outputDim
        self.training = training
        self.nLow = nLow
        self.CELoss = CELOSS
        self.dropout_rate=0.2
    # def _hourglass(self, inputs, n, numOut, name='hourglass'):
    #     """ Hourglass Module
    #     Args:
    #         inputs	: Input Tensor
    #         n		: Number of downsampling step
    #         numOut	: Number of Output Features (channels)
    #         name	: Name of the block
    #     """
    #     with tf.name_scope(name):
    #         # Upper Branch
    #         up = [None] * self.nModules
    #
    #         up[0] = Residual(inputs, numOut, name='up_0')
    #         for i in range(1,self.nModules):
    #             up[i] = Residual(up[i - 1], numOut, name='up_%d' % i)
    #         # Lower Branch
    #         low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
    #         low1 = [None] * self.nModules
    #         low1[0] = Residual(low_, numOut, name='low_0')
    #         for j in range(1,self.nModules):
    #             low1[j] = Residual(low1[j - 1], numOut, name='low_%d' % j)
    #
    #
    #         if n > 0:
    #             low2 = [None]
    #             low2[0] = self._hourglass(low1[self.nModules - 1], n - 1, numOut, name='low_2')
    #         else:
    #             low2 = [None] * self.nModules
    #             low2[0] = Residual(low1[self.nModules - 1], numOut, name='low2_0')
    #             for k in range(1, self.nModules):
    #                 low2[k] = Residual(low2[k - 1], numOut, name='low2_%d' % k)
    #
    #         low3 = [None] * self.nModules
    #         low3[0] = Residual(low2[-1], numOut, name='low_3_0')
    #         for p in range(1,self.nModules):
    #             low3[p] = Residual(low3[p - 1], numOut, name='low3_%d' % j)
    #         up_2 = tf.image.resize_nearest_neighbor(low3[-1], tf.shape(low3[-1])[1:3] * 2, name='upsampling')
    #
    #         return tf.add_n([up_2, up[-1]], name='out_hg')



    def _graph_hourglass(self, inputs,reuse=False):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """

        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128',reuse=reuse)
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1',reuse=reuse)
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = self._residual(pool1, numOut=int(self.nFeats / 2), name='r2',reuse=reuse)
                r3 = self._residual(r2, numOut=self.nFeats, name='r3',reuse=reuse)
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            res = [[None] * self.nModules] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack

            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = self._hourglass(r3, self.nLow, self.nFeats, 'hourglass0',reuse=reuse)
                    # res[0][0] = Residual(hg[0], self.nFeats,name="res0")
                    # for mod in range(1,self.nModules):
                    #     res[0][mod] = Residual(res[0][mod - 1], self.nFeats,name="res%d" % mod)
                    # ll[0] = self._conv_bn_relu(res[0][self.nModules - 1], self.nFeats, 1, 1, 'VALID', name='conv')
                    # ll[0] = self._conv_bn_relu(hg[0], self.nFeats, 1, 1, 'VALID', name='conv0',reuse=reuse)
                    drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training, name='dropout')
                    ll[0] = self._conv_bn_relu(drop[0], self.nFeats, 1, 1, 'VALID', name='conv',reuse=reuse)
                    ll_[0] = self._conv(ll[0], self.nFeats, 1, 1, 'VALID', 'll0',reuse=reuse)

                    out[0] = self._conv(ll[0], self.partnum, 1, 1, 'VALID', name = 'out0',reuse=reuse)
                    out_[0] = self._conv(out[0], self.nFeats, 1, 1, 'VALID', 'out_0',reuse=reuse)
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge0')

                for i in range(1, self.nStack - 1):
                    with tf.variable_scope('stage_' + str(i), reuse=reuse):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeats, 'hourglass',reuse=reuse)
                            # res[i][0] = Residual(hg[i], self.nFeats, name="res0")
                            # for mod in range(1, self.nModules):
                            #     res[i][mod] = Residual(res[i][mod - 1], self.nFeats, name="res%d" % mod)
                            # ll[i] = self._conv_bn_relu(res[i][self.nModules - 1], self.nFeats, 1, 1, 'VALID', name='conv')
                            ll[i] = self._conv_bn_relu(hg[i], self.nFeats, 1, 1, 'VALID', name='conv',reuse=reuse)
                            # drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                            #                             name='dropout',reuse=reuse)
                            # ll[i] = self._conv_bn_relu(hg[i], self.nFeats, 1, 1, 'VALID', name='conv',reuse=reuse)
                            ll_[i] = self._conv(ll[i], self.nFeats, 1, 1, 'VALID', 'll',reuse=reuse)

                            out[i] = self._conv(ll[i], self.partnum, 1, 1, 'VALID', name = 'out',reuse=reuse)
                            out_[i] = self._conv(out[i], self.nFeats, 1, 1, 'VALID', 'out_',reuse=reuse)
                            sum_[i] = tf.add_n([out_[i],  [i - 1], ll_[i]], name='merge')
                with tf.variable_scope('stage_' + str(self.nStack - 1), reuse=reuse):
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeats, 'hourglass',reuse=reuse)
                        # res[self.nStack - 1][0] = Residual(hg[self.nStack - 1], self.nFeats)
                        # for mod in range(1, self.nModules):
                        #     res[self.nStack - 1][mod] = Residual(res[self.nStack - 1][mod - 1], self.nFeats, name="res%d" % mod)
                        #
                        # ll[self.nStack - 1] = self._conv_bn_relu(res[self.nStack - 1][self.nModules - 1], self.nFeats, 1, 1, 'VALID', 'conv')
                        # ll[self.nStack - 1] = self._conv_bn_relu(hg[self.nStack - 1], self.nFeats, 1, 1, 'VALID', 'conv',reuse=reuse)
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                                  training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeats, 1, 1, 'VALID', 'conv',reuse=reuse)

                        out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.partnum, 1, 1, 'VALID', 'out',reuse=reuse)
            if self.CELoss:
                return tf.stack(out, axis= 1 , name = 'final_output')
            else:
                return out
            #return tf.stack(out, axis= 1 , name = 'final_output')

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv',reuse=False):
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
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):

                # Kernel for convolution, Xavier Initialisation
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                    [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
                conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
                return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu',reuse=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                    [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
                conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
                norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                    is_training=self.training)
                return norm

    def _conv_block(self, inputs, numOut, name='conv_block',reuse=False):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                    conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
                return conv_3

    def _skip_layer(self, inputs, numOut, name='skip_layer',reuse=False):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                if inputs.get_shape().as_list()[3] == numOut:
                    return inputs
                else:
                    conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                    return conv

    def _residual(self, inputs, numOut, name='residual_block',reuse=False):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                convb = self._conv_block(inputs, numOut,reuse=reuse)
                skipl = self._skip_layer(inputs, numOut,reuse=reuse)

                return tf.add_n([convb, skipl], name='res_block')

    def _hourglass(self, inputs, n, numOut, name='hourglass',reuse=False):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name, reuse=reuse):
            with tf.name_scope(name):
                # Upper Branch
                up_1 = self._residual(inputs, numOut, name='up_1',reuse=reuse)
                # Lower Branch
                low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
                low_1 = self._residual(low_, numOut, name='low_1',reuse=reuse)

                if n > 0:
                    low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2',reuse=reuse)
                else:
                    low_2 = self._residual(low_1, numOut, name='low_2',reuse=reuse)

                low_3 = self._residual(low_2, numOut, name='low_3',reuse=reuse)
                up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')

                return tf.add_n([up_2, up_1], name='out_hg')