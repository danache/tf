
import tensorflow as tf

from hg_models.layers.Residual import Residual


class HourglassModel():
    def __init__(self, nFeats=256, nStack=4, nModules=1, outputDim=14,nLow=4,training=True
                 ):

        self.nStack = nStack
        self.nFeats = nFeats
        self.nModules = nModules
        self.partnum = outputDim
        self.training = training
        self.nLow = nLow

    def _hourglass(self, inputs, n, numOut, name='hourglass'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up = [None] * self.nModules

            up[0] = Residual(inputs, numOut, name='up_0')
            for i in range(1,self.nModules):
                up[i] = Residual(up[i - 1], numOut, name='up_%d' % i)
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low1 = [None] * self.nModules
            low1[0] = Residual(low_, numOut, name='low_0')
            for j in range(1,self.nModules):
                low1[j] = Residual(low1[j - 1], numOut, name='low_%d' % j)


            if n > 0:
                low2 = [None]
                low2[0] = self._hourglass(low1[self.nModules - 1], n - 1, numOut, name='low_2')
            else:
                low2 = [None] * self.nModules
                low2[0] = Residual(low1[self.nModules - 1], numOut, name='low2_0')
                for k in range(1, self.nModules):
                    low2[k] = Residual(low2[k - 1], numOut, name='low2_%d' % k)

            low3 = [None] * self.nModules
            low3[0] = Residual(low2[-1], numOut, name='low_3_0')
            for p in range(1,self.nModules):
                low3[p] = Residual(low3[p - 1], numOut, name='low3_%d' % j)
            up_2 = tf.image.resize_nearest_neighbor(low3[-1], tf.shape(low3[-1])[1:3] * 2, name='upsampling')

            return tf.add_n([up_2, up[-1]], name='out_hg')

    def lin(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm

    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """

        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = Residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = Residual(pool1, numOut=int(self.nFeats / 2), name='r2')
                r3 = Residual(r2, numOut=self.nFeats, name='r3')
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            res = [[None] * self.nModules] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack

            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = self._hourglass(r3, self.nLow, self.nFeats, 'hourglass')
                    res[0][0] = Residual(hg[0], self.nFeats,name="res0")
                    for mod in range(1,self.nModules):
                        res[0][mod] = Residual(res[0][mod - 1], self.nFeats,name="res%d" % mod)
                    ll[0] = self._conv_bn_relu(res[0][self.nModules - 1], self.nFeats, 1, 1, 'VALID', name='conv')

                    ll_[0] = self._conv(ll[0], self.nFeats, 1, 1, 'VALID', 'll')

                    out[0] = self._conv(ll[0], self.partnum, 1, 1, 'VALID', 'out')
                    out_[0] = self._conv(out[0], self.nFeats, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, self.nStack - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeats, 'hourglass')
                        res[i][0] = Residual(hg[i], self.nFeats, name="res0")
                        for mod in range(1, self.nModules):
                            res[i][mod] = Residual(res[i][mod - 1], self.nFeats, name="res%d" % mod)
                        ll[i] = self._conv_bn_relu(res[i][self.nModules - 1], self.nFeats, 1, 1, 'VALID', name='conv')

                        ll_[i] = self._conv(ll[i], self.nFeats, 1, 1, 'VALID', 'll')

                        out[i] = self._conv(ll[i], self.partnum, 1, 1, 'VALID', 'out')
                        out_[i] = self._conv(out[i], self.nFeats, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.name_scope('stage_' + str(self.nStack - 1)):
                    hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeats, 'hourglass')
                    res[self.nStack - 1][0] = Residual(hg[self.nStack - 1], self.nFeats)
                    for mod in range(1, self.nModules):
                        res[self.nStack - 1][mod] = Residual(res[self.nStack - 1][mod - 1], self.nFeats, name="res%d" % mod)

                    ll[self.nStack - 1] = self._conv_bn_relu(res[self.nStack - 1][self.nModules - 1], self.nFeats, 1, 1, 'VALID', 'conv')

                    out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.partnum, 1, 1, 'VALID', 'out')
            return out

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
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
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)

            return norm

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
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
"""
    def generateModel(self):
        generate_time = time.time()
        #####生成训练数据
        train_data = DataGenerator(imgdir=self.train_img_path, label_dir=self.train_label_path,
                                   out_record=self.train_record,
                                   batch_size=self.batchSize, scale=False, is_valid=False, name="train",
                                   color_jitting=False,flipping=False,)

        self.train_num = train_data.getN()
        train_img, train_heatmap = train_data.getData()
        #####生成验证数据
        if self.valid_img_path:
            valid_data = DataGenerator(imgdir=self.valid_img_path, label_dir=self.valid_label_path,
                                       out_record=self.valid_record,
                                       batch_size=self.batchSize, scale=False, is_valid=True, name="valid")
            valid_img, valid_ht = valid_data.getData()
            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batchSize)
        print('data generate in ' + str(int(time.time() - generate_time)) + ' sec.')
        print("train num is %d, valid num is %d" % (self.train_num, self.valid_num))



        with tf.device(self.gpu[0]):
            self.train_output = self._graph_hourglass(train_img)
            if self.valid_img_path:
                self.valid_output = self._graph_hourglass(valid_img, reuse=True)
            with tf.name_scope('loss'):
                self.loss = self.MSE(output=self.train_output.outputs, target=train_heatmap, is_mean=True)

        if self.valid_img_path:
            with tf.name_scope('acc'):
                self.acc = accuracy_computation(self.valid_output.outputs, valid_ht, batch_size=self.batchSize,
                                              nstack=self.nStack)
            with tf.name_scope('test'):
                for i in range(self.partnum):
                    tf.summary.scalar(self.joints[i],self.acc[i], collections = ['test'])
        with tf.name_scope('train'):
            tf.summary.scalar("train_loss", self.loss, collections=['train'])

        with tf.name_scope('Session'):
            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('lr'):
            self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                 staircase=True, name='learning_rate')
        with tf.name_scope('rmsprop'):
            self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        with tf.name_scope('minimizer'):
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)

        # with tf.device(self.cpu):
        #     with tf.name_scope('training'):
        #         tf.summary.scalar('loss', self.loss, collections=['train'])
        #         tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        #     with tf.name_scope('summary'):
        #         for i in range(len(self.joints)):
        #             tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])


        self.merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')


    def training_init(self, nEpochs=10, saveStep=10):

        with tf.name_scope('Session'):
            for i in self.gpu:
                with tf.device(i):
                    self._init_weight()
                    self.saver = tf.train.Saver()
                    if self.cont:
                        self.saver.restore(self.Session, self.cont)
                    self.train(nEpochs, saveStep)


    def predict(self, img_dir,load,thresh=0.3):
        #if os.path.isdir(img_dir):
        self._init_weight()
        x = tf.placeholder(dtype= tf.float32, shape= (None, 256, 256, 3), name='test_img')
        predict = self._graph_hourglass(x)
        with self.tf.Graph().as_default():
            with tf.device(self.gpu[0]):
                self._init_weight()
                self.saver = tf.train.Saver()
                if self.cont:
                    self.saver.restore(self.Session, load)
            with tf.name_scape('predction'):
                pred_sigmoid = tf.nn.sigmoid(predict[:, self.nStack - 1],
                                                     name='sigmoid_final_prediction')
                pred_final = predict[:, self.HG.nStack - 1]
        pred_lst = []
        if os.path.isdir(img_dir):
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    pred_lst.append(os.path.join(root, file))
        else:
            pred_lst.append(img_dir)
        for img_file in pred_lst:
            img = cv2.imread(img_file)
            board_w , board_h = img.shape[1], img.shape[0]
            resize = 256
            if board_h < board_w:
                newsize = (resize, board_h * resize // board_w)
            else:
                newsize = (board_w * resize // board_h, resize)


            tmp = cv2.resize(img, newsize)
            new_img = np.zeros((resize, resize, 3))
            if (tmp.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                up = np.int((resize - tmp.shape[0]) * 0.5)
                down = np.int((resize + tmp.shape[0]) * 0.5)
                new_img[up:down, :, :] = tmp
            elif (tmp.shape[1] < resize):
                left = np.int((resize - tmp.shape[1]) * 0.5)
                right = np.int((resize + tmp.shape[1]) * 0.5)
                new_img[:, left:right, :] = tmp
            hg = self.Session.run(pred_sigmoid,
                                     feed_dict={img: np.expand_dims(new_img / 255, axis=0)})
            j = np.ones(shape=(self.partnum, 2)) * -1

            for i in range(len(j)):
                idx = np.unravel_index(hg[0, :, :, i].argmax(), (64, 64))
                if hg[0, idx[0], idx[1], i] > thresh:
                    j[i] = np.asarray(idx) * 256 / 64
                    cv2.circle(img_res, center=tuple(j[i].astype(np.int))[::-1], radius=5, color=self.color[i],
                                   thickness=-1)

"""