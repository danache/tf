import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from models.layers.Residual import Residual

from dataGenerator.datagen import DataGenerator
import opt
class HourglassModel():
    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=4, outputDim=14, batch_size=2, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train=None, logdir_test=None, tiny=True, modif=True, name='tiny_hourglass',img_path="",label_path=""
                 ,out_record="/home/dan/tf/test.tfrecords",train_num=1000):
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
        self.img_path = img_path
        self.label_path =label_path
        self.out_record = out_record
        self.train_num = 1000

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
            #end = out[0]
            print("out len = %s" %(len(out)))
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

    def train(self):
        with tf.device('/cpu:0'):
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            startTime = time.time()
            n_epoch = 200
            n_step_epoch = int(self.train_num / self.batchSize)
            n_step = n_epoch * n_step_epoch
            print_freq = 1

            with tf.device('/cpu:0'):

                data_Generator = DataGenerator(imgdir=self.img_path, label_dir=self.label_path, out_record=self.out_record,
                                               batch_size=self.batchSize,scale=False)
                img, heatmap = data_Generator.getData()

            with tf.device(self.gpu):
                # with tf.name_scope('inputs'):
                #     img = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x_train')
                #     heatmap = tf.placeholder(tf.float32, [None, opt.nStack, 64, 64, opt.partnum], name='y_train')
                # TODO : Implement weighted loss function
                # NOT USABLE AT THE MOMENT
                # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
                inputTime = time.time()
                print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
                self.output = self._graph_hourglass(img).outputs
                tf.summary.image('heatmap',self.output,14)
                graphTime = time.time()
                print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')

                with tf.name_scope('loss'):
                    self.loss = self.MSE(output=self.output, target=heatmap, is_mean=True)
                tf.summary.scalar("loss", self.loss)
                tf.summary.histogram("histogram", self.loss)
                lossTime = time.time()
                print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')

            merged = tf.summary.merge_all()
            train_writer=tf.summary.FileWriter("./train",sess.graph)

            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)



            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                     staircase=True, name='learning_rate')

            with tf.device(self.gpu):
                with tf.name_scope('rmsprop'):
                    self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)

            step = 0
            for epoch in range(n_epoch):
                train_loss, train_acc, n_batch = 0, 0, 0
                print("epoch % d" % epoch  )
                for s in range(n_step_epoch):
                    ## You can also use placeholder to feed_dict in data after using
                    # val, l = sess.run([x_train_batch, y_train_batch])
                    # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
                    # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
                    print("epoch % d step : % d" % (epoch,s))
                    err = sess.run([self.loss])

                    step += 1
                    train_loss += err

                    n_batch += 1
                train_writer.add_summary(epoch, train_loss)
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    print("Epoch %d : Step %d-%d of %d took %fs" % (
                    epoch, step, step + n_step_epoch, n_step, time.time() - startTime))
                    print("   train loss: %f" % (train_loss / n_batch))
                    print("   train acc: %f" % (train_acc / n_batch))

                    # test_loss, test_acc, n_batch = 0, 0, 0
                    # for _ in range(int(len(y_test) / batch_size)):
                    #     err, ac = sess.run([cost_test, acc_test])
                    #     test_loss += err;
                    #     test_acc += ac;
                    #     n_batch += 1
                    # print("   test loss: %f" % (test_loss / n_batch))
                    # print("   test acc: %f" % (test_acc / n_batch))

                if (epoch + 1) % (print_freq * 50) == 0:
                    print("Save model " + "!" * 10)
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, "model_test_advanced.ckpt")

            coord.request_stop()
            coord.join(threads)
            sess.close()

"""
    def generate_model(self):

        startTime = time.time()
        sess = tf.InteractiveSession()

        print('CREATE MODEL:')
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                img = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x_train')
                heatmap = tf.placeholder(tf.float32, [None,opt.nStack, 64, 64, opt.partnum], name='y_train')
            # TODO : Implement weighted loss function
            # NOT USABLE AT THE MOMENT
            # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')
            self.output = self._graph_hourglass(img).outputs
            graphTime = time.time()
            print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
            
            with tf.name_scope('loss'):
                self.loss = self.MSE(output=self.output, target=heatmap, is_mean=True)
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

"""