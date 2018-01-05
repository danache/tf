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
                 val_label="",train_label="",partnum=14,human_decay=0.96,val_batch_num=10000,beginepoch=0,CELOSS=False
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

        self.h_decay = tf.Variable(1.,  trainable=False,dtype=tf.float32,)

        self.last_learning_rate = tf.Variable(self.learn_r,trainable=False )

        generate_train_done_time = time.time()
        print('train data generate in ' + str(int(generate_train_done_time - generate_time)) + ' sec.')
        print("train num is %d" % (self.train_num))
        self.global_step = 0
        with tf.name_scope('lr'):
            # self.lr = tf.train.exponential_decay(self.learn_r, self.train_step, self.lr_decay_step,
            #                                    self.lr_decay,name='learning_rate')

            self.lr = get_lr(self.last_learning_rate, self.global_step, self.lr_decay_step,
                             self.lr_decay, self.h_decay, name='learning_rate')
        #####random show an image
        c = tf.constant(np.arange(0, self.batch_size))
        shuff = tf.random_shuffle(c)
        n = shuff[0]
        n = tf.cast(n,tf.int32 )

        ###list for multi gpu
        tower_grads = []

        self.train_img_set = [None] * len(self.gpu)
        self.train_hm_set = [None]* len(self.gpu)
        self.train_center_set = [None]* len(self.gpu)
        self.train_scale_set = [None]* len(self.gpu)
        self.train_name_set = [None]* len(self.gpu)
        self.train_output_set = [None] * len(self.gpu)
        self.totalloss = [None] * len(self.gpu)
        self.loss_dict = [[]] * len(self.gpu)

        self.repeat = [[]] * len(self.gpu)
        self.trans_heatmap = [None] * len(self.gpu)

        self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.loss = []
        ###define multi GPU loss
        flag= False
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in range(len(self.gpu)):
                self.train_img_set[i], self.train_hm_set[i], self.train_center_set[i],\
                self.train_scale_set[i], self.train_name_set[i] = train_data.getData()
                with tf.device(("/gpu:%d" % self.gpu[i])):
                    with tf.name_scope('gpu_%d' % (self.gpu[i])) as scope:
                        self.train_output_set[i] = self.model(self.train_img_set[i],reuse=flag)
                        flag=True
                        with tf.name_scope('loss'):
                            with tf.device(self.cpu):
                                #allloss = tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output.outputs)

                                if self.celoss:

                                    for i in range(self.nstack):
                                        self.repeat[i].append(self.train_hm_set[i])
                                    self.trans_heatmap[i] = tf.stack(self.repeat[i], axis=1)
                                    self.loss.append(tf.reduce_mean(
                                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_output_set[i],
                                                                                labels=self.trans_heatmap[i]),
                                        name='cross_entropy_loss'))
                                else:

                                    for nsta in range(len( self.train_output_set[i])):

                                        self.loss_dict[i].append(tf.losses.mean_squared_error(labels=self.train_hm_set[i],
                                                                                           predictions=
                                                                                           self.train_output_set[i][nsta]))
                                    self.totalloss[i] = tf.add_n(self.loss_dict[i])

                                self.loss.append(self.totalloss[i])

                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('loss_%d' % (i), self.loss[i], collections=['train'])
                                with tf.name_scope('heatmap'):

                                    im = self.train_img_set[i][n, :, :, :]
                                    im = tf.expand_dims(im, 0)

                                    tf.summary.image(name=('origin_img_%d'%(i)), tensor=im, collections=['train'])
                                    tout = []
                                    tgt = []
                                    for joint in range(self.partnum):
                                        if self.celoss:
                                            hm = self.train_output_set[i][n,-1, :, :, joint]
                                            tmp = self.train_output_set[i][n,-1, :, :, joint]
                                        else:
                                            hm = self.train_output_set[i][-1][n, :, :, joint]
                                            tmp = self.train_output_set[i][-1][n, :, :, joint]
                                        hm = tf.expand_dims(hm, -1)
                                        hm = tf.expand_dims(hm, 0)
                                        gt = self.train_hm_set[i][n, :, :, joint]

                                        gt = tf.expand_dims(gt, -1)
                                        gt = tf.expand_dims(gt, 0)
                                        #gt = gt * 255
                                        tf.summary.image('ground_truth_%s_%d' % (self.joints[joint],0), tensor=gt,
                                                         collections=['train'])
                                        tf.summary.image('heatmp_%s_%d' % (self.joints[joint],0), hm, collections=['train'])


                                        tout.append(tf.cast(tf.equal(tf.reduce_max(tmp), tmp), tf.float32))
                                        tmp2 = self.train_hm_set[i][n, :, :, joint]
                                        tgt.append(tf.cast(tf.equal(tf.reduce_max(tmp2), tmp2), tf.float32))
                                    train_gt = tf.add_n(tgt)
                                    train_gt = tf.expand_dims(train_gt, 0)
                                    train_gt = tf.expand_dims(train_gt, -1)
                                    train_hm = tf.add_n(tout)
                                    train_hm = tf.expand_dims(train_hm, 0)
                                    train_hm = tf.expand_dims(train_hm, -1)
                                    tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
                                    tf.summary.image('train_heatmp', train_hm, collections=['train'])

                        grads = self.rmsprop.compute_gradients(loss= self.loss[i])
                        tower_grads.append(grads)
        grads_ = self.average_gradients(tower_grads)
        self.apply_gradient_op = self.rmsprop.apply_gradients(grads_)

        # with tf.name_scope('heatmap'):
        #
        #     im = train_img[n, :, :, :]
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
        #     tf.summary.image('train_ground_truth', tensor=train_gt, collections=['train'])
        #     tf.summary.image('train_heatmp', train_hm, collections=['train'])


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
        #     print(len(self.train_output))
        #     for nsta in range(len(self.train_output)):
        #         print(self.train_output[nsta])
        #         self.loss_dict.append(tf.losses.mean_squared_error(labels=self.train_heatmap,predictions=self.train_output[nsta]))
        #     self.loss = tf.add_n(self.loss_dict)

        with tf.name_scope('training'):
            for gpus_n in range(len(self.gpu)):

                tf.summary.scalar('loss %d'%gpus_n , self.loss[gpus_n], collections=['train'])
        ######
        if self.valid_record:
            valid_data = self.valid_record
            self.valid_img,  self.valid_heatmap, self.valid_center, self.valid_scale, self.valid_name = valid_data.getData()

            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batch_size)
            #self.valid_output = self._graph_hourglass(self.valid_img)
            self.valid_output = self.model(self.valid_img,reuse=True)

            generate_valid_done_time = time.time()
            print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
            print("valid num is %d" % (self.valid_num))

            with tf.name_scope('val_heatmap'):

                val_im = self.valid_img[n, :, :, :]
                val_im = tf.expand_dims(val_im, 0)

                tf.summary.image(name=('origin_valid_img' ), tensor=val_im, collections=['test'])

                for joint in range(self.partnum):
                    if self.celoss:
                        val_hm = self.valid_output[n,-1, :, :, joint]
                    else:
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
        if self.celoss:
            self.train_coord = reverseFromHt(self.train_output_set[0][:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
                                             num_joint=self.partnum,
                                             scale=self.train_scale_set[0], center=self.train_center_set[0], res=[64, 64])
            self.valid_coord = reverseFromHt(self.valid_output[:,-1,:], nstack=self.nstack, batch_size=self.batch_size,
                                             num_joint=self.partnum,
                                             scale=self.valid_scale, center=self.valid_center, res=[64, 64])
        else:
            self.train_coord =reverseFromHt(self.train_output_set[0][-1], nstack=self.nstack, batch_size=self.batch_size, num_joint=self.partnum,
                                            scale=self.train_scale_set[0], center=self.train_center_set[0], res=[64, 64])

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

        n_step_epoch = int(self.train_num / (self.batch_size * len(self.gpu)))
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
                        ([self.apply_gradient_op,self.train_merged,self.lr,self.train_coord,self.train_name_set[0]],
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
