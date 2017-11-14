import time
import tensorflow as tf
from eval import MSE
import numpy as np
import sys
import os
import cv2
import tensorlayer as tl

from eval.eval import accuracy_computation

class train_class():
    def __init__(self, model, nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='tiny_hourglass', train_record="",
                valid_record="",save_model_dir="",resume="",gpu=[0],
                 partnum=14,
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
        self.gpu = [("/gpu:%d" % i) for i in gpu]
        self.cpu = '/cpu:0'
        self.model = model
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]

    def generateModel(self):
        generate_time = time.time()
        train_data = self.train_record
        self.train_num = train_data.getN()
        train_img, train_heatmap = train_data.getData()

        self.train_output = self.model(train_img)
        generate_train_done_time = time.time()
        print('train data generate in ' + str(int(generate_train_done_time - generate_time)) + ' sec.')
        print("train num is %d" % (self.train_num))
        #####生成验证数据
        if self.valid_record:
            valid_data = self.valid_record
            valid_img, valid_ht = valid_data.getData()
            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batch_size)
            self.valid_output = self.model(valid_img, reuse=True)
            generate_valid_done_time = time.time()
            print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
            print("valid num is %d" % (self.valid_num))

        for i in self.gpu:
            with tf.device(i):
                with tf.name_scope('loss'):
                    allloss, stack_loss, part_loss = MSE.MSE(output=self.train_output.outputs, target=train_heatmap,
                                        nstack=4, partnum=self.partnum,is_mean=True)
                    self.loss = allloss
                    self.stack_loss = stack_loss
                    self.part_loss = part_loss
                if self.valid_record:
                    with tf.name_scope('acc'):
                        self.acc = accuracy_computation(self.valid_output.outputs, valid_ht, batch_size=self.batch_size,
                                                      nstack=self.nstack)
                    with tf.name_scope('test'):
                        for i in range(self.partnum):
                            tf.summary.scalar(self.joints[i],self.acc[i], collections = ['test'])

                with tf.name_scope('Session'):
                    with tf.name_scope('steps'):
                        self.train_step = tf.Variable(0, name='global_step', trainable=False)
                with tf.name_scope('lr'):
                    self.lr = tf.train.exponential_decay(self.learn_r, self.train_step, self.lr_decay_step, self.lr_decay,
                                   staircase=True, name='learning_rate')
                with tf.name_scope('rmsprop'):
                    self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                with tf.name_scope('minimizer'):
                    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(self.update_ops):
                        self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)

        with tf.device(self.cpu):
            with tf.name_scope('training'):
                print(type(self.loss))
                tf.summary.scalar('loss', self.loss, collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
            with tf.name_scope('summary'):
                for i in range(self.nstack):
                    tf.summary.scalar("stack%d"%i, self.stack_loss[i], collections=['train'])
                for j in range(self.partnum):
                    tf.summary.scalar(self.joints[i], self.part_loss[j], collections=['train'])

        self.train_merged = tf.summary.merge_all('train')
        self.valid_merge = tf.summary.merge_all('test')

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.Session.run(tf.global_variables_initializer())

        self.Session.run(tf.local_variables_initializer())
        tl.layers.initialize_global_variables(self.Session)
        print("init done")

    def training_init(self, nEpochs=10, saveStep=10):
        with tf.name_scope('Session'):
            for i in self.gpu[0]:
                self._init_weight()
                self.saver = tf.train.Saver()
                if self.resume:
                    self.saver.restore(self.Session, self.resume)
                self.train(nEpochs, saveStep)

    def train(self, nEpochs=10, saveStep=10):
        #####参数定义
        self.resume = {}
        self.resume['accur'] = []
        self.resume['loss'] = []
        self.resume['err'] = []

        n_step_epoch = int(self.train_num / self.batch_size)
        self.train_writer = tf.summary.FileWriter(self.logdir_train, self.Session.graph)
        self.valid_writer = tf.summary.FileWriter(self.logdir_valid)

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.Session)
        self.Session.run(init)

        for epoch in range(nEpochs):
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

                if n_batch % saveStep == 0:
                    _,lo ,sta_lo, part_lo,summary= self.Session.run\
                        ([self.train_rmsprop,self.loss,self.stack_loss,self.part_loss,self.train_merged])

                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()
                else:
                    _, lo = self.Session.run([self.train_rmsprop, self.loss])
                loss += lo
                avg_cost += lo / n_step_epoch
            epochfinishTime = time.time()
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')

            # with tf.name_scope('save'):
            #     self.saver.save(self.Session, os.path.join(self.save_dir,self.name + '_' + str(epoch + 1)+
            #                                                "_"+(str(avg_cost)[:6])))
            accuracy_array = np.zeros([1,14])
            self.resume['loss'].append(loss)

            #####valid
            if self.valid_record:
                for i in range(self.validIter):#self.validIter
                    accuracy_pred = self.Session.run([self.acc])
                    #print(np.array(accuracy_pred).shape)
                    accuracy_array += np.array(accuracy_pred, dtype=np.float32) / self.validIter
                print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)))[:6], )
                self.resume['accur'].append(accuracy_pred)
                self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                valid_summary = self.Session.run([self.valid_merge])

                self.valid_writer.add_summary(valid_summary[0],epoch)
                self.valid_writer.flush()
        coord.request_stop()
        coord.join(threads)
        self.Session.close()
        print('Training Done')
