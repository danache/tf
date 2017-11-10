import time
import tensorflow as tf
import numpy as np
import sys
import os

import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from models.layers.Residual import Residual
from dataGenerator.datagen import DataGenerator
from eval.eval import accuracy_computation
import opt
os.environ["CUDA_VISIBLE_DEVICES"]='1'
class HourglassModel():
    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=4, outputDim=14, batch_size=32, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train="./log/train/", logdir_test="./log/test/", tiny=True, modif=True, name='tiny_hourglass',
                 train_img_path="", train_label_path="", train_record="",
                 valid_img_path="", valid_label_path="", valid_record="",
                 model_dir=""
                 ):
        """ Initializer
        Args:
            nStack				: number of stacks (stage/Hourglass modules)
            nFeat				: number of feature channels on conv layers
            nLow				: number of downsampling (pooling) per module
            outputDim			: number of output Dimension (16 for MPII)
            batch_size			: size of training/testing Batch
            dro_rate			: Rate of neurons disabling for Dropout Layers
            lear_rate			: Learning Rate starting value
            decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
            decay_step			: Step to apply decay
            dataset			: Dataset (class DataGenerator)
            training			: (bool) True for training / False for prediction
            w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
            tiny				: (bool) Activate Tiny Hourglass
            modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
            name				: name of the model
        """
        self.nStack = nStack
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.tiny = tiny
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.modif = modif
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:1'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head',
                       'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        self.train_img_path = train_img_path
        self.train_label_path = train_label_path
        self.train_record = train_record

        self.valid_img_path = valid_img_path
        self.valid_label_path = valid_label_path
        self.valid_record = valid_record
        self.model_dir = model_dir
        self.train_num = 1000
        self.valid_num = 1000

    def get_input(self):
        """ Returns Input (Placeholder) Tensor
        Image Input :
            Shape: (None,256,256,3)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.img

    def get_output(self):
        """ Returns Output Tensor
        Output Tensor :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.output

    def get_label(self):
        """ Returns Label (Placeholder) Tensor
        Image Input :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.gtMaps

    def get_loss(self):
        """ Returns Loss Tensor
        Image Input :
            Shape: (1,)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.loss

    def get_saver(self):
        """ Returns Saver
        /!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
        Warning:
            Be sure to build the model first
        """
        return self.saver

    def hourglass(self, data, n, f, name=""):

        # Upper Branch
        up_1 = Residual(data, f, f, name='%s_up_1' % (name))
        # Lower Branch
        low_ = tl.layers.MaxPool2d(data, (2, 2), strides=(2, 2), name='%s_pool1' % (name))
        low_1 = Residual(low_, f, f, name='%s_low_1' % (name))

        if n > 0:
            low_2 = self.hourglass(low_1, n - 1, f, name='%s_low_2' % (name))
        else:
            low_2 = Residual(low_1, f, f, name='%s_low_2' % (name))

        low_3 = Residual(low_2, f, f, name='%s_low_3' % (name))
        up_2 = tl.layers.UpSampling2dLayer(low_3, size=[2, 2], is_scale=True, method=1, name="%s_Upsample" % (name))

        return tl.layers.ElementwiseLayer(layer=[up_1, up_2],
                                          combine_fn=tf.add, name="%s_add_n" % (name))

    def lin(self, data, numOut, name=None):
        with tf.variable_scope(name) as scope:
            conv1 = conv_2d(data, numOut, filter_size=(1, 1), strides=(1, 1),
                            name='conv1')
            bn1 = tl.layers.BatchNormLayer(conv1, act=tf.nn.relu, name="bn1")

            return bn1

    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.device(self.gpu):
            with tf.name_scope('model'):
                with tf.name_scope('preprocessing'):
                    data = tl.layers.InputLayer(inputs, name='input')
                    with tf.name_scope('train'):
                        tf.summary.histogram('data', data.outputs, collections=['train'])

                    conv1 = conv_2d(data, 64, filter_size=(6, 6), strides=(2, 2), padding="SAME", name="conv1")

                    bn1 = tl.layers.BatchNormLayer(conv1, name="bn1", act=tf.nn.relu)
                    with tf.name_scope('train'):
                        tf.summary.histogram("conv1_bn/weight", conv1.all_params[0], collections=['train'])
                        tf.summary.histogram('conv1_bn', bn1.outputs, collections=['train'])
                    r1 = Residual(bn1, 64, 128, name="Residual1")
                    with tf.name_scope('train'):
                        tf.summary.histogram('residual1', r1.outputs, collections=['train'])

                    pool = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

                    r2 = Residual(pool, 128, 128, name="Residual2")
                    with tf.name_scope('train'):
                        tf.summary.histogram('residual2', r2.outputs, collections=['train'])
                    r3 = Residual(r2, 128, opt.nFeats, name="Residual3")
                    with tf.name_scope('train'):
                        tf.summary.histogram('residual3', r3.outputs, collections=['train'])
                        # return r3
                # Storage Table

                out = []
                inter = r3
                with tf.name_scope('stacks'):
                    for i in range(self.nStack):
                        with tf.name_scope('stage_%d' % (i)):
                            hg = self.hourglass(inter, n=4, f=opt.nFeats, name="stage_%d_hg" % (i))

                            r1 = Residual(hg, opt.nFeats, opt.nFeats, name="stage_%d_Residual1" % (i))
                            ll = self.lin(r1, opt.nFeats, name="stage_%d_lin1" % (i))
                            tmpout = conv_2d(ll, opt.partnum, filter_size=(1, 1), strides=(1, 1),
                                             name="stage_%d_tmpout" % (i))
                            out.append(tmpout)
                            if i < self.nStack - 1:
                                ll_ = conv_2d(ll, opt.nFeats, filter_size=(1, 1), strides=(1, 1),
                                              name="stage_%d_ll_" % (i))
                                tmpOut_ = conv_2d(tmpout, opt.nFeats, filter_size=(1, 1), strides=(1, 1),
                                                  name="stage_%d_tmpOut_" % (i))
                                inter = tl.layers.ElementwiseLayer(layer=[inter, ll_, tmpOut_],
                                                                   combine_fn=tf.add, name="stage_%d_add_n" % (i))

                # end = out[0]
                end = tl.layers.StackLayer(out, axis=1, name='final_output')
                # end = tl.layers.StackLayer([out])
                return end

    def MSE(self, output, target, is_mean=False):
        print(output.get_shape())
        print(target.get_shape())

        with tf.name_scope("mean_squared_error_loss"):
            if output.get_shape().ndims == 5:  # [batch_size, n_feature]
                if is_mean:
                    mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3, 4]))
                else:
                    mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3, 4]))
                return mse
            else:
                raise Exception("Unknow dimension")


    def train(self, nEpochs=10, saveStep=500):
        self.resume = {}
        self.resume['accur'] = []
        self.resume['loss'] = []
        self.resume['err'] = []

        generate_time = time.time()

        train_data = DataGenerator(imgdir=self.train_img_path, label_dir=self.train_label_path,
                                   out_record=self.train_record,
                                   batch_size=self.batchSize, scale=False, is_valid=False, name="train")


        self.train_num = train_data.getN()
        train_img, train_heatmap = train_data.getData()
        n_epoch = nEpochs
        n_step_epoch = int(self.train_num / self.batchSize)

        n_step = n_epoch * n_step_epoch
        print_freq = 1
        if self.valid_img_path:
            valid_data = DataGenerator(imgdir=self.valid_img_path, label_dir=self.valid_label_path,
                                       out_record=self.valid_record,
                                       batch_size=self.batchSize, scale=False, is_valid=True, name="valid")
            valid_img, valid_ht = valid_data.getData()
            self.valid_num = valid_data.getN()
            validIter = int(self.valid_num / self.batchSize)



        print('data generate in ' + str(int(time.time() - generate_time)) + ' sec.')

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with tf.device(self.gpu):


            self.Session.run(tf.global_variables_initializer())

            self.Session.run(tf.local_variables_initializer())
            tl.layers.initialize_global_variables(self.Session)

            self.train_output = self._graph_hourglass(train_img)
            if self.valid_img_path:
                self.valid_output = self._graph_hourglass(valid_img)

            graphTime = time.time()

            print('Graph build in ' + str(int(generate_time - graphTime)) + ' sec.')
            with tf.name_scope('loss'):
                # self.loss = tf.contrib.losses.mean_squared_error(self.output,heatmap)
                self.loss = self.MSE(output=self.train_output.outputs, target=train_heatmap, is_mean=True)  # +\

                # self.loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output.outputs, labels= heatmap), name = 'cross_entropy_loss')
                # tf.contrib.layers.l2_regularizer(0.01)(self.output.all_params[0]) + tf.contrib.layers.l2_regularizer(0.01)(self.output.all_params[2])

            if self.valid_img_path:
                with tf.name_scope('acc'):
                    self.acc = accuracy_computation(self.valid_output,valid_ht,batch_size=self.batchSize,nstack=self.nStack)
                with tf.name_scope('valid'):
                    tf.summary.scalar("valid_loss", self.acc, collections=['valid'])

            with tf.name_scope('train'):
                tf.summary.scalar("train_loss", self.loss, collections=['train'])


            merged = tf.summary.merge_all('train')
            valid_merge = tf.summary.merge_all('valid')
            train_writer = tf.summary.FileWriter("./log/train.log", self.Session.graph)
            valid_writer = tf.summary.FileWriter("./log/valid.log")

            lossTime = time.time()
            print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')

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

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=self.Session)
            self.Session.run(init)

            #with tf.device(self.gpu):

            with tf.name_scope('rmsprop'):
                self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

            for epoch in range(nEpochs):
                epochstartTime = time.time()
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                loss = 0
                avg_cost = 0.
                for n_batch in range(n_step_epoch):
                    percent = ((n_batch + 1) / n_step_epoch) * 100
                    num = np.int(20 * percent / 100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                    sys.stdout.write(
                        '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                      :4] + '%' + ' -cost: ' + str(
                            loss)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                    sys.stdout.flush()

                    if n_batch % saveStep == 0:
                        _, lo, summary = self.Session.run([self.train_rmsprop, self.loss, merged])
                        train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                        train_writer.flush()
                    else:
                        _, lo = self.Session.run([self.train_rmsprop, self.loss])
                    loss += lo
                    avg_cost += lo / n_step_epoch
                epochfinishTime = time.time()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                    int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                    ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(self.model_dir,self.name + '_' + str(epoch + 1)))
                accuracy_array = np.array([0.0] * len(opt.partnum))
                self.resume['loss'].append(loss)
                if self.valid_img_path:
                    for i in range(validIter):
                        accuracy_pred = self.Session.run([accuracy_pred])
                        accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
                    print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%')
                    self.resume['accur'].append(accuracy_pred)
                    self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                    valid_summary = self.Session.run([valid_merge])
                    valid_writer.add_summary(valid_summary)
                    valid_writer.flush()



            coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            self.Session.close()
            print('Training Done')

    def training_init(self, nEpochs=10, saveStep=500, dataset=None, load=None):
        """ Initialize the training
        Args:
            nEpochs		: Number of Epochs to train
            epochSize		: Size of one Epoch
            saveStep		: Step to save 'train' summary (has to be lower than epochSize)
            dataset		: Data Generator (see generator.py)
            load			: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()

                if load is not None:
                    self.saver.restore(self.Session, load)
                # try:
                #	self.saver.restore(self.Session, load)
                # except Exception:
                #	print('Loading Failed! (Check README file for further information)')
                self.train(nEpochs, saveStep)

    def _define_saver_summary(self, summary=True):
        """ Create Summary and Saver
        Args:
            logdir_train		: Path to train summary directory
            logdir_test		: Path to test summary directory
        """
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        # else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver()
                #if summary:
                #     with tf.device(self.gpu):
                #         self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                #         self.test_summary = tf.summary.FileWriter(self.logdir_test)
                #         # self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        t_start = time.time()
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')
