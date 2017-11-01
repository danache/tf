import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from models.layers.Residual import Residual


import opt
class HourglassModel():
    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=4, outputDim=14, batch_size=16, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train=None, logdir_test=None, tiny=True, modif=True, name='tiny_hourglass'):
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
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head',
                       'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

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

    def hourglass(self,data,n,f,name=""):
        print("hourglassd name %s" % (name))

        # Upper Branch
        up_1 = Residual(data, f,f, name='%s_up_1'%(name))
        # Lower Branch
        low_ =  tl.layers.MaxPool2d(data, (2,2),strides=(2,2),name='%s_pool1'%(name) )
        low_1 = Residual(low_, f,f, name='%s_low_1'%(name))

        if n > 0:
            low_2 = self.hourglass(low_1, n - 1, f, name='%s_low_2'%(name))
        else:
            low_2 = Residual(low_1, f,f, name='%s_low_2'%(name))

        low_3 = Residual(low_2, f,f, name='%s_low_3'%(name))
        up_2 = tl.layers.UpSampling2dLayer(low_3, size=[2,2], is_scale = True,method=1,name="%s_Upsample"%(name))

        return tl.layers.ElementwiseLayer(layer=[up_1,up_2],
                               combine_fn=tf.add, name="%s_add_n"%(name))

    def lin(self,data, numOut, name=None):
        with tf.variable_scope(name) as scope:
            conv1 = conv_2d(data, numOut, filter_size=(1, 1), strides=(1, 1),
                            name='conv1')
            bn1 = tl.layers.BatchNormLayer(conv1, act=tf.nn.relu, name="bn1" )

            return bn1

    def _accuracy_computation(self):
        """ Computes accuracy tensor
        """
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(
                self._accur(self.output[:, self.nStack - 1, :, :, i], self.gtMaps[:, self.nStack - 1, :, :, i],
                            self.batchSize))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of max position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.arg_max(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])
    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))

    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                data = tl.layers.InputLayer(inputs, name='input')
                conv1 = conv_2d(data, 64, filter_size=(6, 6), strides=(2, 2),  padding="SAME", name="conv1")
                bn1 = tl.layers.BatchNormLayer(conv1, name="bn1", act=tf.nn.relu)


                r1 = Residual(bn1,64, 128,name="Residual1")

                pool = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

                r2 = Residual(pool,128, 128,name="Residual2")

                r3 = Residual(r2,128, opt.nFeats,name="Residual3")

            # Storage Table

            out = []
            inter = r3
            with tf.name_scope('stacks'):
                for i in range(self.nStack):
                    with tf.name_scope('stage_%d' % (i)):
                        hg = self.hourglass(inter, n=4, f=opt.nFeats,name="stage_%d_hg"%(i))
                        r1 = Residual(hg,opt.nFeats,opt.nFeats,name="stage_%d_Residual1"%(i))
                        ll = self.lin(r1,opt.nFeats,name="stage_%d_lin1" % (i))
                        tmpout = conv_2d(ll,opt.partnum,filter_size=(1, 1), strides=(1, 1),name="stage_%d_tmpout" % (i))
                        out.append(tmpout)
                        if i < self.nStack - 1:
                            ll_ = conv_2d(ll,opt.nFeats,filter_size=(1, 1), strides=(1, 1),name="stage_%d_ll_"%(i))
                            tmpOut_ = conv_2d(tmpout,opt.nFeats,filter_size=(1, 1), strides=(1, 1),name="stage_%d_tmpOut_" % (i))
                            inter = tl.layers.ElementwiseLayer(layer=[inter,ll_,tmpOut_],
                                   combine_fn=tf.add, name="stage_%d_add_n"%(i))

            end = tl.layers.StackLayer(out, axis=1, name='final_output')
            #end = tl.layers.StackLayer([out])
            return end
    def MSE(self,output, target, is_mean=False):
        with tf.name_scope("mean_squared_error_loss"):
            if output.get_shape().ndims == 5:  # [batch_size, n_feature]
                if is_mean:
                    mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3,4]))
                else:
                    mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3,4]))
                return mse
            else:
                raise Exception("Unknow dimension")


    def generate_model(self):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
                self.img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')
                # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
                self.gtMaps = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))
            # TODO : Implement weighted loss function
            # NOT USABLE AT THE MOMENT
            # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
            self.output = self._graph_hourglass(self.img).outputs
            graphTime = time.time()
            print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
            
            with tf.name_scope('loss'):
                self.loss = self.MSE(output=self.output, target=self.gtMaps, is_mean=True)
            lossTime = time.time()
            print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')
        with tf.device(self.cpu):
            with tf.name_scope('accuracy'):
                self._accuracy_computation()
            accurTime = time.time()
            print('---Acc : Done (' + str(int(abs(accurTime - lossTime))) + ' sec.)')
            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                     staircase=True, name='learning_rate')
            lrTime = time.time()
            print('---LR : Done (' + str(int(abs(accurTime - lrTime))) + ' sec.)')
        with tf.device(self.gpu):
            with tf.name_scope('rmsprop'):
                self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            optimTime = time.time()
            print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')
            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
            minimTime = time.time()
            print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')
        self.init = tf.global_variables_initializer()
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')
        with tf.device(self.cpu):
            with tf.name_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
            with tf.name_scope('summary'):
                for i in range(len(self.joints)):
                    tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])
        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')
        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime

