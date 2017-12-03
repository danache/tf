import time
import tensorflow as tf
import numpy as np
import os
import cv2
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from hg_models.layers.Residual import Residual
from dataGenerator.datagen import DataGenerator
from eval.eval import accuracy_computation

class HourglassModel():
    def __init__(self, nFeat=256, nStack=4, nModules=1, outputDim=14,
                 ):

        self.nStack = nStack
        self.nFeats = nFeat
        self.nModules = nModules
        self.partnum = outputDim


    def hourglass(self, data, n, f, name="",reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # Upper Branch
            up_1 = Residual(data, f, f, name='%s_up_1' % (name),reuse=reuse)
            # Lower Branch
            low_ = tl.layers.MaxPool2d(data, (2, 2), strides=(2, 2), name='%s_pool1' % (name))
            low_1 = Residual(low_, f, f, name='%s_low_1' % (name),reuse=reuse)

            if n > 0:
                low_2 = self.hourglass(low_1, n - 1, f, name='%s_low_2' % (name),reuse=reuse)
            else:
                low_2 = Residual(low_1, f, f, name='%s_low_2' % (name),reuse=reuse)

            low_3 = Residual(low_2, f, f, name='%s_low_3' % (name),reuse=reuse)
            up_2 = tl.layers.UpSampling2dLayer(low_3, size=[2, 2], is_scale=True, method=1, name="%s_Upsample" % (name))

            return tl.layers.ElementwiseLayer(layer=[up_1, up_2],
                                              combine_fn=tf.add, name="%s_add_n" % (name))

    def lin(self, data, numOut, name=None,reuse=False):
        with tf.variable_scope(name,reuse=reuse) as scope:
            conv1 = conv_2d(data, numOut, filter_size=(1, 1), strides=(1, 1),
                            name='conv1')
            bn1 = tl.layers.BatchNormLayer(conv1, act=tf.nn.relu, name="bn1")

            return bn1

    def _graph_hourglass(self, inputs,reuse=False):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
        """

        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            data = tl.layers.InputLayer(inputs, name='input')
            with tf.name_scope('train'):
                tf.summary.histogram('data', data.outputs, collections=['train'])

            conv1 = conv_2d(data, 64, filter_size=(7, 7), strides=(2, 2), padding="SAME", name="conv1")
            #conv1 (128,128,64)
            bn1 = tl.layers.BatchNormLayer(conv1, name="bn1", act=tf.nn.relu)
            # with tf.name_scope('train'):
            #     tf.summary.histogram("conv1_bn/weight", conv1.all_params[0], collections=['train'])
            #     tf.summary.histogram('conv1_bn', bn1.outputs, collections=['train'])
            r1 = Residual(bn1, 64, 128, name="Residual1",reuse=reuse)
            #r1 (128,128,128)
            # with tf.name_scope('train'):
            #     tf.summary.histogram('residual1', r1.outputs, collections=['train'])

            pool = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

            r2 = Residual(pool, 128, 128, name="Residual2",reuse=reuse)
            # r2 (64,64,128)

            r3 = Residual(r2, 128, self.nFeats, name="Residual3",reuse=reuse)
            # 64,64,256
                    # return r3
            # Storage Table

        out = []

        hg = [None] * self.nStack
        residual=[None] * self.nStack
        ll = [None] * self.nStack
        fc_out = [None] * self.nStack
        c_1 = [None] * self.nStack
        c_2 = [None] * self.nStack
        sum_ = [None] * self.nStack


        with tf.variable_scope("stack", reuse=reuse):
            with tf.name_scope('stage_0'):
                hg[0] = self.hourglass(r3, n=4, f=self.nFeats, name="stage_0_hg", reuse=reuse)

                residual[0] = Residual(hg[0], self.nFeats, self.nFeats, name="stage_0_Residual",reuse=reuse)
                ll[0] = self.lin( residual[0], self.nFeats, name="stage_0_lin1" , reuse=reuse)
                fc_out[0] = conv_2d(ll[0], self.partnum, filter_size=(1, 1), strides=(1, 1),
                                 name="stage_0_out" )
                out.append(fc_out[0])

                c_1[0] = conv_2d(ll[0], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                              name="stage_0_conv1")

                c_2[0] = conv_2d(c_1[0], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                  name="stage_0_conv2")
                sum_[0] = tl.layers.ElementwiseLayer(layer=[r3, c_1[0], c_2[0]],
                                                        combine_fn=tf.add, name="stage_0_add_n" )

            for i in range(1,self.nStack-1):
                with tf.name_scope('stage_%d' % (i)):
                    hg[i] = self.hourglass(sum_[i-1], n=4, f=self.nFeats, name="stage_%d_hg" % (i),reuse=reuse)

                    residual[i] = Residual(hg[i], self.nFeats, self.nFeats, name="stage_%d_Residual"% (i), reuse=reuse)
                    ll[i] = self.lin(residual[i], self.nFeats, name="stage_%d_lin"% (i), reuse=reuse)
                    fc_out[i] = conv_2d(ll[i], self.partnum, filter_size=(1, 1), strides=(1, 1),
                                        name="stage_%d_out"% (i))
                    out.append(fc_out[i])

                    c_1[i] = conv_2d(ll[i], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                     name="stage_%d_conv1"% (i))

                    c_2[i] = conv_2d(c_1[i], self.nFeats, filter_size=(1, 1), strides=(1, 1),
                                     name="stage_%d_conv2"% (i))
                    sum_[i] = tl.layers.ElementwiseLayer(layer=[sum_[i-1], c_1[i], c_2[i]],
                                                         combine_fn=tf.add, name="stage_%d_add_n"% (i))
            with tf.name_scope('stage_%d' % (self.nStack - 1)):
                hg[self.nStack - 1] = self.hourglass(sum_[self.nStack-2], n=4, f=self.nFeats,
                                                     name="stage_%d_hg" % (self.nStack-1), reuse=reuse)

                residual[self.nStack - 1] = Residual(hg[self.nStack - 1], self.nFeats, self.nFeats, name="stage_%d_Residual"% (self.nStack-1), reuse=reuse)
                ll[self.nStack - 1] = self.lin(residual[self.nStack - 1], self.nFeats, name="stage_%d_lin1"% (self.nStack-1), reuse=reuse)
                fc_out[self.nStack - 1] = conv_2d(ll[self.nStack - 1], self.partnum, filter_size=(1, 1), strides=(1, 1),
                                    name="stage_%d_out"% (self.nStack-1))
                out.append(fc_out[self.nStack - 1])
        # end = out[0]
        end = tl.layers.StackLayer(out, axis=1, name='final_output')
        # end = tl.layers.StackLayer([out])
        return end

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