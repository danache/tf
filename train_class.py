import os
import sys
import time

import numpy as np
import tensorflow as tf

from eval.ht2coord import getjointcoord

from tools.keypoint_eval import getScore
from tools.keypoint_eval import load_annotations
from tools.lr import get_lr
from tools.img_tf import *
class train_class():
    def __init__(self, model, nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='tiny_hourglass', train_record="",valid_record="",save_model_dir="",resume="",gpu=[0],
                 val_label="",train_label="",partnum=14,human_decay=0.96,val_batch_num=10000,beginepoch=0
                 ):
        self.batch_size = batch_size
        self.nstack = nstack
        self.learn_r = learn_rate

        self.lr_decay = decay
        self.lr_decay_step = decay_step
        self.logdir_train = logdir_train
        self.logdir_valid = logdir_valid
        self.name = name
        self.resume = resume
        self.train_record = train_record
        self.valid_record = valid_record
        self.save_dir = save_model_dir
        self.gpu = gpu
        self.cpu = '/cpu:0'
        self.model = model
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]
        self.val_label = val_label
        self.mae = tf.Variable(0, trainable=False, dtype=tf.float32,)
        self.human_decay = human_decay
        self.beginepoch = beginepoch
        self.val_batch_num=val_batch_num
        self.train_mae = tf.Variable(0, trainable=False, dtype=tf.float32, )
        self.train_label=train_label
        self.training = True
    def average_gradients(self,tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # if g:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # if flag:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def generateModel(self):
        generate_time = time.time()
        train_data = self.train_record
        self.train_num = train_data.getN()
        train_img, self.train_heatmap, self.train_center, self.train_scale,self.train_name = train_data.getData()

        #self.train_output = self.model(train_img)

        self.h_decay = tf.Variable(1.,  trainable=False,dtype=tf.float32,)

        self.last_learning_rate = tf.Variable(self.learn_r,trainable=False )

        generate_train_done_time = time.time()
        print('train data generate in ' + str(int(generate_train_done_time - generate_time)) + ' sec.')
        print("train num is %d" % (self.train_num))
        self.global_step = 0


        tower_grads = []

        with tf.name_scope('lr'):
            # self.lr = tf.train.exponential_decay(self.learn_r, self.train_step, self.lr_decay_step,
            #                                    self.lr_decay,name='learning_rate')

            self.lr = get_lr(self.last_learning_rate, self.global_step, self.lr_decay_step,
                             self.lr_decay, self.h_decay, name='learning_rate')


        self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        flag = False
        self.loss = []
        c = tf.constant(np.arange(0, self.batch_size))
        shuff = tf.random_shuffle(c)
        n = shuff[0]
        n = tf.cast(n,tf.int32 )
        # with tf.variable_scope(tf.get_variable_scope()) as vscope:
        #     for i in self.gpu:
        #         with tf.device(("/gpu:%d" % i)):
        #             with tf.name_scope('gpu_%d' % (i)) as scope:
        #
        #                 self.train_output = self.model(train_img,reuse=flag)
        #
        #                 flag = True
        #                 with tf.name_scope('loss'):
        #                     with tf.device(self.cpu):
        #                         #allloss = tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output.outputs)
        #
        #                         total_loss = 0
        #                         for nsta in range(self.nstack):
        #                             total_loss += tf.reduce_mean(tf.square(tf.subtract(self.train_heatmap,
        #                                                                                                      self.train_output[
        #                                                                                                          nsta].outputs)), [1, 2, 3])
        #                                 # tf.losses.mean_squared_error(labels=self.train_heatmap[:,nsta,:],
        #                                 #                                        predictions=self.train_output[nsta].outputs)
        #                         # total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_output.outputs,
        #                         #                                                                    labels=self.train_heatmap),
        #                         #                            name='cross_entropy_loss')
        #
        #                         self.loss.append(total_loss)
        #
        #                         with tf.name_scope('training'):
        #                             # print(type(self.loss))
        #                             tf.summary.scalar('loss_%d' % (i), self.loss[i], collections=['train'])
        #                         with tf.name_scope('heatmap'):
        #
        #                             im = train_img[n, :, :, :]
        #                             im = tf.expand_dims(im, 0)
        #
        #                             tf.summary.image(name=('origin_img_%d'%(i)), tensor=im, collections=['train'])
        #                             tout = []
        #                             tgt = []
        #                             for joint in range(self.partnum):
        #                                 hm = self.train_output[-1].outputs[n, :, :, joint]
        #                                 hm = tf.expand_dims(hm, -1)
        #                                 hm = tf.expand_dims(hm, 0)
        #                                 #hm = hm * 255
        #                                 gt = self.train_heatmap[n, :, :, joint]
        #
        #                                 gt = tf.expand_dims(gt, -1)
        #                                 gt = tf.expand_dims(gt, 0)
        #                                 #gt = gt * 255
        #                                 tf.summary.image('ground_truth_%s_%d' % (self.joints[joint],0), tensor=gt,
        #                                                  collections=['train'])
        #                                 tf.summary.image('heatmp_%s_%d' % (self.joints[joint],0), hm, collections=['train'])
        #                                 tmp = self.train_output[-1].outputs[n,  :, :, joint]
        #                                 tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
        #                                 tmp2 = self.train_heatmap[n, :, :, joint]
        #                                 tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))
        #                             train_gt = tf.add_n(tgt)
        #
        #                             train_gt = tf.expand_dims(train_gt, 0)
        #                             train_gt = tf.expand_dims(train_gt, -1)
        #                             train_hm = tf.add_n(tout)
        #
        #                             train_hm = tf.expand_dims(train_hm, 0)
        #                             train_hm = tf.expand_dims(train_hm, -1)
        #                             tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
        #                             tf.summary.image('train_heatmp', train_hm, collections=['train'])
        #
        #
        #                 grads = self.rmsprop.compute_gradients(loss= self.loss[i])
        #                 tower_grads.append(grads)
        # grads_ = self.average_gradients(tower_grads)
        # self.apply_gradient_op = self.rmsprop.apply_gradients(grads_)

        ###########
        #self.train_output = self._graph_hourglass(train_img)
        self.train_output = self.model(train_img)


        with tf.name_scope('heatmap'):

            im = train_img[n, :, :, :]
            im = tf.expand_dims(im, 0)

            tf.summary.image(name=('origin_img'), tensor=im, collections=['train'])
            tout = []
            tgt = []
            for joint in range(self.partnum):
                hm = self.train_output[-1][n, :, :, joint]
                hm = tf.expand_dims(hm, -1)
                hm = tf.expand_dims(hm, 0)
                #hm = hm * 255
                gt = self.train_heatmap[n, :, :, joint]

                gt = tf.expand_dims(gt, -1)
                gt = tf.expand_dims(gt, 0)
                #gt = gt * 255
                tf.summary.image('ground_truth_%s' % (self.joints[joint]), tensor=gt,
                                 collections=['train'])
                tf.summary.image('heatmp_%s_%d' % (self.joints[joint],0), hm, collections=['train'])
                tmp = self.train_output[-1][n,  :, :, joint]
                tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
                tmp2 = self.train_heatmap[n, :, :, joint]
                tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))
            train_gt = tf.add_n(tgt)

            train_gt = tf.expand_dims(train_gt, 0)
            train_gt = tf.expand_dims(train_gt, -1)
            train_hm = tf.add_n(tout)

            train_hm = tf.expand_dims(train_hm, 0)
            train_hm = tf.expand_dims(train_hm, -1)
            tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
            tf.summary.image('train_heatmp', train_hm, collections=['train'])


        self.loss = 0
        for nsta in range(len(self.train_output)):
            self.loss += tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output[nsta])

        with tf.name_scope('training'):
            # print(type(self.loss))
            tf.summary.scalar('loss' , self.loss, collections=['train'])
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_output.outputs, labels=self.train_heatmap),
        #                name='cross_entropy_loss')
        self.apply_gradient_op = self.rmsprop.minimize(self.loss)
        ######
        if self.valid_record:
            valid_data = self.valid_record
            self.valid_img,  self.valid_heatmap, self.valid_center, self.valid_scale, self.valid_name = valid_data.getData()

            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batch_size)
            #self.valid_output = self._graph_hourglass(self.valid_img)
            self.valid_output = self.model(self.valid_img)

            generate_valid_done_time = time.time()
            print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
            print("valid num is %d" % (self.valid_num))

            with tf.name_scope('val_heatmap'):

                val_im = self.valid_img[n, :, :, :]
                val_im = tf.expand_dims(val_im, 0)

                tf.summary.image(name=('origin_valid_img' ), tensor=val_im, collections=['test'])

                for joint in range(self.partnum):
                    val_hm = self.valid_output[-1][n, :, :, joint]
                    val_hm = tf.expand_dims(val_hm, -1)
                    val_hm = tf.expand_dims(val_hm, 0)
                    #val_hm = val_hm * 255
                    val_gt = self.valid_heatmap[n,  :, :, joint]

                    val_gt = tf.expand_dims(val_gt, -1)
                    val_gt = tf.expand_dims(val_gt, 0)
                    #val_gt = val_gt * 255
                    tf.summary.image('valid_ground_truth_%s' % (self.joints[joint]), tensor=val_gt,
                                     collections=['test'])
                    tf.summary.image('valid_heatmp_%s' % (self.joints[joint]), val_hm, collections=['test'])

            # with tf.name_scope('acc'):
            #     self.acc = accuracy_computation(self.valid_output.outputs, valid_ht, batch_size=self.batch_size,
            #                                     nstack=self.nstack)
            # with tf.name_scope('test'):
            #     for i in range(self.partnum):
            #         tf.summary.scalar(self.joints[i], self.acc[i], collections=['test'])

        with tf.device(self.cpu):

            with tf.name_scope('training'):
                # print(type(self.loss))
                #tf.summary.scalar('loss_0', self.loss[0], collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
                tf.summary.scalar("MAE", self.train_mae, collections=['train'])

            # with tf.name_scope('summary'):
            #     for i in range(self.nstack):
            #         tf.summary.scalar("stack%d"%i, self.stack_loss[i], collections=['train'])
            #     for j in range(self.partnum):
            #         tf.summary.scalar(self.joints[i], self.part_loss[j], collections=['train'])
            with tf.name_scope('MAE'):
                tf.summary.scalar("MAE", self.mae, collections=['test'])

        self.train_coord =reverseFromHt(self.train_output[-1], nstack=self.nstack, batch_size=self.batch_size, num_joint=self.partnum,
                                        scale=self.train_scale, center=self.train_center, res=[64, 64])

        self.valid_coord = reverseFromHt(self.valid_output[-1], nstack=self.nstack, batch_size=self.batch_size,
                                         num_joint=self.partnum,
                                         scale=self.valid_scale, center=self.valid_center, res=[64, 64])

        self.train_merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.Session.run(tf.global_variables_initializer())

        self.Session.run(tf.local_variables_initializer())
        print("init done")

    def training_init(self, nEpochs=10, valStep=3000,showStep=10):
        with tf.name_scope('Session'):

            self._init_weight()
            self.saver = tf.train.Saver()
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())

            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.Session)
            self.Session.run(init)

            if self.resume:
                print("resume from"+self.resume)
                self.saver.restore(self.Session, self.resume)
            self.train(nEpochs, valStep,showStep)

    def train(self, nEpochs=10, valStep = 3000,showStep=10 ):
        #best_val = open("./best_val.txt", "w")
        best_model_dir = ""
        best_val = -99999
        #####参数定义
        self.resume = {}
        self.resume['accur'] = []
        self.resume['loss'] = []
        self.resume['err'] = []
        return_dict = dict()
        return_dict['error'] = None
        return_dict['warning'] = []
        return_dict['score'] = None
        anno = load_annotations(self.val_label, return_dict)

        n_step_epoch = int(self.train_num / self.batch_size)
        self.train_writer = tf.summary.FileWriter(self.logdir_train, self.Session.graph)
        self.valid_writer = tf.summary.FileWriter(self.logdir_valid)


        last_lr = self.learn_r
        hm_decay = 1
        valStep = min(valStep, n_step_epoch -100)
        if self.validIter < self.val_batch_num:
            val_batch_num = self.validIter
        else:
            val_batch_num = self.val_batch_num
        for epoch in range(self.beginepoch, nEpochs):
            self.global_step += 1
            epochstartTime = time.time()
            print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
            loss = 0
            avg_cost = 0.

            for n_batch in range(n_step_epoch):#n_step_epoch


                percent = ((n_batch + 1) / n_step_epoch) * 100
                num = np.int(20 * percent / 100)
                tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                sys.stdout.write(
                    '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                  :4] + '%' + ' -cost: ' + str(
                        loss)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                sys.stdout.flush()

                if n_batch % showStep == 0:
                    _,summary,last_lr,train_coord,train_name= self.Session.run\
                        ([self.apply_gradient_op,self.train_merged,self.lr,self.train_coord,self.train_name],
                         feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()

                    train_predictions = dict()
                    train_predictions['image_ids'] = []
                    train_predictions['annos'] = dict()
                    train_predictions = getjointcoord(train_coord, train_name, train_predictions)

                    train_return_dict = dict()
                    train_return_dict['error'] = None
                    train_return_dict['warning'] = []
                    train_return_dict['score'] = None
                    train_anno = load_annotations(self.train_label, train_return_dict)
                    train_score = getScore(train_predictions, train_anno, train_return_dict)

                    tmp = self.train_mae.assign(train_score)
                    _ = self.Session.run(tmp)


                else:
                    _, last_lr = self.Session.run([self.apply_gradient_op, self.lr],
                                             feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                hm_decay = 1.

                if (n_batch+1) % valStep == 0:

                    if self.valid_record:
                        val_begin = time.time()

                        valid_predictions = dict()
                        valid_predictions['image_ids'] = []
                        valid_predictions['annos'] = dict()
                        val_begin_time = time.time()

                        for i in range(100):  # self.validIter
                            val_percent = ((i + 1) / val_batch_num) * 100
                            val_num = np.int(20 * val_percent / 100)
                            val_tToEpoch = int((time.time() - val_begin) * (100 - val_percent) / (val_percent))

                            val_cord,  val_name = self.Session.run(
                                [ self.valid_coord, self.valid_name,]
                                )

                            # print(np.array(accuracy_pred).shape)
                            valid_predictions = getjointcoord(val_cord,  val_name,  valid_predictions)
                            sys.stdout.write(
                                '\r valid {0}>'.format("=" * val_num) + "{0}>".format(" " * (20 - val_num)) + '||' + str(percent)[
                                                                                                  :4][:4] +
                                '%' + ' -cost: ' +
                                ' -timeToEnd: ' + str(val_tToEpoch) + ' sec.')
                            sys.stdout.flush()
                        print("val done in" + str(time.time() - val_begin_time))
                        score = getScore(valid_predictions, anno, return_dict)
                        tmp = self.mae.assign(score)
                        _ = self.Session.run(tmp)
                        if score > best_val:
                            best_val = score
                            best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch) +
                                                          "_" + str(n_batch) + "_" + (str(score)[:8]))
                            print("get lower loss, save at " + best_model_dir)
                            with tf.name_scope('save'):
                                self.saver.save(self.Session, best_model_dir)
                            hm_decay = 1.

                        else:
                            #print("now val loss is not best, restore model from" + best_model_dir)
                            #self.saver.restore(self.Session, best_model_dir)
                            hm_decay = self.human_decay

                        valid_summary = self.Session.run([self.valid_merge])

                        self.valid_writer.add_summary(valid_summary[0], epoch * n_step_epoch + n_batch)
                        self.valid_writer.flush()

            if epoch % 4 == 0:
                model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch) +
                                              "_" + "base")
                print("epoch %d , save at "%epoch + model_dir)
                with tf.name_scope('save'):
                    self.saver.save(self.Session, model_dir)
            epochfinishTime = time.time()
            # if epoch % 5 == 0:
            #     hm_decay = self.human_decay
            # else:
            #     hm_decay = 1.
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')

            self.resume['loss'].append(loss)

            #####valid


        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')

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
                r1 = self._residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128

                r2 = self._residual(pool1, numOut=int(256 / 2), name='r2')
                r3 = self._residual(r2, numOut=256, name='r3')
            # Storage Table
            hg = [None] * self.nstack
            ll = [None] * self.nstack
            ll_ = [None] * self.nstack
            drop = [None] * self.nstack
            out = [None] * self.nstack
            out_ = [None] * self.nstack
            sum_ = [None] * self.nstack
            
            with tf.name_scope('stacks'):
                with tf.name_scope('stage_0'):
                    hg[0] = self._hourglass(r3, 4, 256, 'hourglass')
                    ll[0] = self._residual(hg[0], numOut=256, name='Residual_0')
                    ll_[0] = self._conv(ll[0], 256, 1, 1, 'VALID', 'll')
  
                    out[0] = self._conv(ll[0], 14, 1, 1, 'VALID', 'out')
                    out_[0] = self._conv(out[0], 256, 1, 1, 'VALID', 'out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, self.nstack - 1):
                    with tf.name_scope('stage_' + str(i)):
                        hg[i] = self._hourglass(sum_[i - 1], 4, 256, 'hourglass')
                        ll[i] = self._residual(hg[i], numOut=256, name='Residual_0')
                        ll_[i] = self._conv(ll[i], 256, 1, 1, 'VALID', 'll')

                        out[i] = self._conv(ll[i], 14, 1, 1, 'VALID', 'out')
                        out_[i] = self._conv(out[i], 256, 1, 1, 'VALID', 'out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.name_scope('stage_' + str(self.nstack - 1)):
                    hg[self.nstack - 1] = self._hourglass(sum_[self.nstack - 2], 4, 256,
                                                          'hourglass')

                    ll[self.nstack - 1] = self._residual(hg[self.nstack - 1], numOut=256, name='Residual_0')

                    out[self.nstack - 1] = self._conv(ll[self.nstack - 1], 14, 1, 1, 'VALID',
                                                              'out')
                return out

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

    def _conv_block(self, inputs, numOut, name='conv_block'):
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

    def _skip_layer(self, inputs, numOut, name='skip_layer'):
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
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)

            return tf.add_n([convb, skipl], name='res_block')

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
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')

            return tf.add_n([up_2, up_1], name='out_hg')