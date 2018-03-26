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

def EnsureDir(dirs):
    if os.path.isdir(dirs):
        return
    else:
        os.mkdir(dirs)

class train_class():
    def __init__(self, model, nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='tiny_hourglass', train_record="",valid_record="",save_model_dir="",resume="",gpu=[0],
                 val_label="",train_label="",partnum=14,human_decay=0.96,val_batch_num=10000,beginepoch=0,CELOSS=False,
                 train_num=0
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
        self.celoss = CELOSS
        self.train_num = train_num




    def average_gradients(self,tower_grads):
        """
        多GPU计算，平均梯度
        Calculate the average gradient for each shared variable across all towers.
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
        print(zip(*tower_grads))
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # if g:
                    # Add 0 dimension to the gradients to represent the tower.
                    if g is None:
                        continue
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)
            # if flag:
            # Average over the 'tower' dimension.
            if len(grads) == 0:
                continue
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
        '''
        #搭建网络及相关,包括tensorboard展示，loss计算等

        '''

        train_data = self.train_record
        self.train_num = train_data.getN()

        self.h_decay = tf.Variable(1., trainable=False, dtype=tf.float32, )

        self.last_learning_rate = tf.Variable(self.learn_r, trainable=False)
        self.global_step = 0
        '''
        随机取得一个batch里的图片作为展示
        '''
        shuff = tf.random_shuffle(tf.constant(np.arange(0, self.nstack)))
        n = tf.cast(shuff[0], tf.int32)

        with tf.name_scope('lr'):
            # self.lr = tf.train.exponential_decay(self.learn_r, self.train_step, self.lr_decay_step,
            #                                    self.lr_decay,name='learning_rate')
            #获取学习率
            self.lr = get_lr(self.last_learning_rate, self.global_step, self.lr_decay_step,
                             self.lr_decay, self.h_decay, name='learning_rate')
        ###list for multi gpu
        tower_grads = []

        self.train_img_lst = []

        self.train_heatmap_lst = []

        self.train_output_set = [None] * len(self.gpu)

        self.totalloss = [None] * len(self.gpu)

        self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.allloss = []
        ###define multi GPU loss
        flag= False

        with tf.variable_scope(tf.get_variable_scope()) as vscope:

            for i in range(len(self.gpu)):
                print("/gpu:%d" % self.gpu[i])
                with tf.device(("/gpu:%d" % self.gpu[i])):
                    with tf.name_scope('gpu_%d' % (self.gpu[i])) as scope:
                        train_img, train_mini, train_heatmap = train_data.TensorflowBatch()
                        self.train_img_lst.append(train_img)

                        self.train_heatmap_lst.append(train_heatmap)
                        #获取输出
                        self.train_output_set[i] = self.model.build(train_img,reuse=flag)
                        flag=True
                        with tf.name_scope('loss'):
                            with tf.device(self.cpu):
                                loss = 0
                                #对每个stack计算loss
                                for nsta in range(self.nstack):
                                    loss += tf.losses.mean_squared_error(labels=self.train_heatmap_lst[i],
                                                                         predictions=
                                                                         self.train_output_set[i][nsta])

                                self.allloss.append(loss)
                        #tensorboard 展示
                        with tf.name_scope('training'):
                            # print(type(self.loss))
                            tf.summary.scalar('loss_%d' % (i), self.allloss[i], collections=['train'])


                        with tf.name_scope('heatmap'):

                            im = self.train_img_lst[i][n, :, :, :]
                            im = tf.expand_dims(im, 0)

                            tf.summary.image(name=('origin_img_%d'%(i)), tensor=im, collections=['train'])
                            tout = []
                            tgt = []
                            for joint in range(self.partnum):

                                hm = self.train_output_set[i][-1][n, :, :, joint]

                                hm = tf.expand_dims(hm, -1)
                                hm = tf.expand_dims(hm, 0)
                                gt = self.train_heatmap_lst[i][n, :, :, joint]

                                gt = tf.expand_dims(gt, -1)
                                gt = tf.expand_dims(gt, 0)
                                gt = gt * 255
                                tf.summary.image('ground_truth_%s_%d' % (self.joints[joint],i), tensor=gt,
                                                 collections=['train'])
                                tf.summary.image('heatmp_%s_%d' % (self.joints[joint],i), hm, collections=['train'])

                                tmp = self.train_output_set[i][-1][n, :, :, joint]
                                tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
                                tmp2 = self.train_heatmap_lst[i][n, :, :, joint]
                                tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))

                            train_gt = tf.add_n(tgt)

                            train_gt = tf.expand_dims(train_gt, 0)

                            train_hm = tf.add_n(tout)

                            train_hm = tf.expand_dims(train_hm, 0)
                            train_hm = tf.expand_dims(train_hm, -1)

                            #train_tmp_img = tf.add(im, tf.stack(repeat, 3))
                            train_gt = tf.expand_dims(train_gt, -1)
                            #tf.summary.image(name=('origin_train_img'), tensor=train_tmp_img, collections=['train'])

                            tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
                            tf.summary.image('train_heatmp', train_hm, collections=['train'])

                        grads = self.rmsprop.compute_gradients(loss= self.allloss[i])
                        tower_grads.append(grads)
        #平均梯度
        grads_ = self.average_gradients(tower_grads)
        #梯度下降
        self.apply_gradient_op = self.rmsprop.apply_gradients(grads_)

        # self.train_output = self.model.build(self.train_img)
        # with tf.name_scope('heatmap'):
        #
        #     im = self.train_img[n, :, :, :]
        #     im = tf.expand_dims(im, 0)
        #
        #     tf.summary.image(name=('origin_img'), tensor=im, collections=['train'])
        #     tout = []
        #     tgt = []
        #     for joint in range(self.partnum):
        #         if self.celoss:
        #             hm = self.train_output[n,-1, :, :, joint]
        #         else:
        #             hm = self.train_output[-1][n, :, :, joint]
        #         hm = tf.expand_dims(hm, -1)
        #         hm = tf.expand_dims(hm, 0)
        #         #hm = hm * 255
        #         gt = self.train_heatmap[n, :, :, joint]
        #
        #         gt = tf.expand_dims(gt, -1)
        #         gt = tf.expand_dims(gt, 0)
        #         #gt = gt * 255
        #         tf.summary.image('ground_truth_%s' % (self.joints[joint]), tensor=gt,
        #                          collections=['train'])
        #         tf.summary.image('heatmp_%s_%d' % (self.joints[joint],0), hm, collections=['train'])
        #         if self.celoss:
        #             tmp = self.train_output[n,-1, :, :, joint]
        #         else:
        #             tmp = self.train_output[-1][n,  :, :, joint]
        #         tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
        #         tmp2 = self.train_heatmap[n, :, :, joint]
        #         tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))
        #     train_gt = tf.add_n(tgt)
        #
        #     train_gt = tf.expand_dims(train_gt, 0)
        #     train_gt = tf.expand_dims(train_gt, -1)
        #     train_hm = tf.add_n(tout)
        #
        #     train_hm = tf.expand_dims(train_hm, 0)
        #     train_hm = tf.expand_dims(train_hm, -1)
        #     tf.summary.image('ground_truth', tensor=train_gt, collections=['train',"test"])
        #     tf.summary.image('heatmp', train_hm, collections=['train',"test"])
        # self.lase_out = self.train_output[-1]
        #
        # self.loss = 0
        # for nsta in range(len(self.train_output)):
        #     self.loss += tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output[nsta])
        # if self.celoss:
        #     repeat = []
        #     for i in range(self.nstack):
        #         repeat.append(self.train_heatmap)
        #     t_heatmap = tf.stack(repeat, axis=1)
        #     self.loss = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_output, labels=t_heatmap),
        #         name='cross_entropy_loss')
        # else:
        #     self.loss_dict = []
        #     for nsta in range(len(self.train_output)):
        #         print(self.train_output[nsta])
        #         self.loss_dict.append(tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output[nsta]))
        #     self.loss = tf.add_n(self.loss_dict)
        # self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        # self.apply_gradient_op = self.rmsprop.minimize(self.loss)
        # with tf.name_scope('training'):
        #     for gpus_n in range(len(self.gpu)):
        #
        #         tf.summary.scalar('loss %d'%gpus_n , self.loss, collections=['train'])
        # ######
        # # if self.valid_record:
        # #     valid_data = self.valid_record
        # #     self.valid_img,  self.valid_heatmap, self.valid_center, self.valid_scale, self.valid_name = valid_data.getData()
        # #
        # #     self.valid_num = valid_data.getN()
        # #     self.validIter = int(self.valid_num / self.batch_size)
        # #     #self.valid_output = self._graph_hourglass(self.valid_img)
        # #     self.valid_output = self.model.build(self.valid_img,reuse=True)
        # #
        # #     generate_valid_done_time = time.time()
        # #     print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
        # #     print("valid num is %d" % (self.valid_num))
        # #
        # #     with tf.name_scope('val_heatmap'):
        # #
        # #         val_im = self.valid_img[n, :, :, :]
        # #         val_im = tf.expand_dims(val_im, 0)
        # #
        # #         tf.summary.image(name=('origin_valid_img' ), tensor=val_im, collections=['test'])
        # #
        # #         for joint in range(self.partnum):
        # #             if self.celoss:
        # #                 val_hm = self.valid_output[n,-1, :, :, joint]
        # #             else:
        # #                 val_hm = self.valid_output[-1][n, :, :, joint]
        # #             val_hm = tf.expand_dims(val_hm, -1)
        # #             val_hm = tf.expand_dims(val_hm, 0)
        # #             #val_hm = val_hm * 255
        # #             val_gt = self.valid_heatmap[n,  :, :, joint]
        # #
        # #             val_gt = tf.expand_dims(val_gt, -1)
        # #             val_gt = tf.expand_dims(val_gt, 0)
        # #             #val_gt = val_gt * 255
        # #             tf.summary.image('valid_ground_truth_%s' % (self.joints[joint]), tensor=val_gt,
        # #                              collections=['test'])
        # #             tf.summary.image('valid_heatmp_%s' % (self.joints[joint]), val_hm, collections=['test'])
        # #
        # #
        # # with tf.device(self.cpu):
        # #
        # #     with tf.name_scope('training'):
        # #         # print(type(self.loss))
        # #         #tf.summary.scalar('loss_0', self.loss[0], collections=['train'])
        # #         tf.summary.scalar('learning_rate', self.lr, collections=['train'])
        # #         tf.summary.scalar("MAE", self.train_mae, collections=['train'])
        # #
        # #     # with tf.name_scope('summary'):
        # #     #     for i in range(self.nstack):
        # #     #         tf.summary.scalar("stack%d"%i, self.stack_loss[i], collections=['train'])
        # #     #     for j in range(self.partnum):
        # #     #         tf.summary.scalar(self.joints[i], self.part_loss[j], collections=['train'])
        # #     with tf.name_scope('MAE'):
        # #         tf.summary.scalar("MAE", self.mae, collections=['test'])
        # # # if self.celoss:
        # # #     self.train_coord = reverseFromHt(self.train_output_set[0][:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
        # # #                                      num_joint=self.partnum,
        # # #                                      scale=self.train_scale_set[0], center=self.train_center_set[0], res=[64, 64])
        # # #     self.valid_coord = reverseFromHt(self.valid_output[:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
        # # #                                      num_joint=self.partnum,
        # # #                                      scale=self.valid_scale, center=self.valid_center, res=[64, 64])
        # # # else:
        # # #     self.train_coord =reverseFromHt(self.train_output_set[0][-1], nstack=self.nstack, batch_size=self.batch_size, num_joint=self.partnum,
        # # #                                     scale=self.train_scale_set[0], center=self.train_center_set[0], res=[64, 64])
        # # #
        # # #     self.valid_coord = reverseFromHt(self.valid_output[-1], nstack=self.nstack, batch_size=self.batch_size,
        # # #                                      num_joint=self.partnum,
        # # #                                      scale=self.valid_scale, center=self.valid_center, res=[64, 64])
        # # if self.celoss:
        # #     self.train_coord = reverseFromHt(self.train_output_set[0][:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
        # #                                      num_joint=self.partnum,
        # #                                      scale=self.train_scale_set[0], center=self.train_center_set[0], res=[64, 64])
        # #     self.valid_coord = reverseFromHt(self.valid_output[:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
        # #                                      num_joint=self.partnum,
        # #                                      scale=self.valid_scale, center=self.valid_center, res=[64, 64])
        # # else:
        # #     self.train_coord =reverseFromHt(self.train_output[-1], nstack=self.nstack, batch_size=self.batch_size, num_joint=self.partnum,
        # #                                     scale=self.train_scale, center=self.train_center, res=[64, 64])
        # #
        # #     self.valid_coord = reverseFromHt(self.valid_output[-1], nstack=self.nstack, batch_size=self.batch_size,
        # #                                      num_joint=self.partnum,
        # #                                      scale=self.valid_scale, center=self.valid_center, res=[64, 64])
        self.train_merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')
    #参数初始化
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
            ##load model
            if self.resume:
                print("resume from"+self.resume)
                self.saver.restore(self.Session, self.resume)
            self.train(nEpochs, valStep,showStep)

    def train(self, nEpochs=10, valStep = 3000,showStep=10 ):
        # self.generator = self.train_record.get_batch_generator()
        # self.valid_gen = self.valid_record.get_batch_generator()
        # #best_val = open("./best_val.txt", "w")
        # best_val = -99999
        # #####参数定义
        # self.resume = {}
        # self.resume['accur'] = []
        # self.resume['loss'] = []
        # self.resume['err'] = []
        # return_dict = dict()
        # return_dict['error'] = None
        # return_dict['warning'] = []
        # return_dict['score'] = None
        # anno = load_annotations(self.val_label, return_dict)

        n_step_epoch = int(self.train_num / (self.batch_size * len(self.gpu)))
        self.train_writer = tf.summary.FileWriter(self.logdir_train, self.Session.graph)
        self.valid_writer = tf.summary.FileWriter(self.logdir_valid)

        last_lr = self.learn_r
        hm_decay = 1
        valStep = min(valStep, n_step_epoch -100)

        val_batch_num = self.val_batch_num
        '''
        开始训练
        '''
        for epoch in range( self.beginepoch,nEpochs):
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
                '''
                如果到了展示的batch，run一下merged进行展示，否则正常进行梯度下降
                '''
                if n_batch % showStep == 0:
                    # _,__,___,summary,last_lr,train_coord, train_name= self.Session.run\
                        # ([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K ,self.train_merged,self.lr,self.train_coord,self.train_name_lst[0]],
                        #  feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _,summary, last_lr,  = self.Session.run \
                        ([self.apply_gradient_op, self.train_merged, self.lr],
                         feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})


                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()

                else:
                    # _, __, ___,last_lr = self.Session.run([self.apply_hg_grads_,self.apply_discrim_grads_,self.update_K , self.lr],
                    #                          feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    _, last_lr = self.Session.run(
                        [self.apply_gradient_op, self.lr],
                        feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})
                #
                hm_decay = 1.

            if epoch % 1 == 0:
                best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch))
                print("epoch "+str(epoch)+", save at " + best_model_dir)
                with tf.name_scope('save'):
                    self.saver.save(self.Session, best_model_dir)

            epochfinishTime = time.time()
            # if epoch % 5 == 0:
            #     hm_decay = self.human_decay
            # else:
            #     hm_decay = 1.
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')

            # self.resume['loss'].append(loss)

            #####valid


        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')
