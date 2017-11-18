import time
import tensorflow as tf
from eval import MSE
import numpy as np
import sys
import os
import tensorlayer as tl

from eval.eval import accuracy_computation
from eval.ht2coord import getjointcoord
from tools.lr import get_lr
from tools.keypoint_eval import getScore
from tools.keypoint_eval import load_annotations
class train_class():
    def __init__(self, model, nstack=4, batch_size=32,learn_rate=2.5e-4, decay=0.96, decay_step=2000,
                 logdir_train="./log/train.log", logdir_valid="./log/test.log",
                name='tiny_hourglass', train_record="",valid_record="",save_model_dir="",resume="",gpu=[0],
                 val_label="",partnum=14,human_decay=0.96,val_batch_num=10000
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
        self.loss = []
        self.stack_loss =[]
        self.part_loss = []
        self.val_batch_num=val_batch_num

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
            flag = False
            for g, _ in grad_and_vars:
                # if g:
                    # Add 0 dimension to the gradients to represent the tower.
                    flag = True
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
        train_img, train_heatmap = train_data.getData()

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


        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in self.gpu:
                with tf.device(("/gpu:%d" % i)):
                    with tf.name_scope('gpu_%d' % ( i)) as scope:

                        self.train_output = self.model(train_img,reuse=flag)
                        flag = True
                        with tf.name_scope('loss'):
                            with tf.device(self.cpu):
                                allloss, stack_loss, part_loss = MSE.MSE(output=self.train_output.outputs, target=train_heatmap,
                                                    nstack=self.nstack, partnum=self.partnum,is_mean=True)

                                self.loss.append(allloss[0])
                                self.stack_loss.append(stack_loss)
                                self.part_loss.append(part_loss)
                                with tf.name_scope('training'):
                                    # print(type(self.loss))
                                    tf.summary.scalar('loss_%d' % (i), self.loss[i], collections=['train'])

                        tf.get_variable_scope().reuse_variables()

                        grads = self.rmsprop.compute_gradients(loss= self.loss[i])
                        tower_grads.append(grads)
        grads_ = self.average_gradients(tower_grads)
        self.apply_gradient_op = self.rmsprop.apply_gradients(grads_)

        if self.valid_record:
            valid_data = self.valid_record
            valid_img, valid_ht,self.valid_size, self.valid_name = valid_data.getData()
            self.valid_num = valid_data.getN()
            self.validIter = int(self.valid_num / self.batch_size)
            self.valid_output = self.model(valid_img, reuse=True)
            generate_valid_done_time = time.time()
            print('train data generate in ' + str(int(generate_valid_done_time - generate_train_done_time )) + ' sec.')
            print("valid num is %d" % (self.valid_num))
            with tf.name_scope('acc'):
                self.acc = accuracy_computation(self.valid_output.outputs, valid_ht, batch_size=self.batch_size,
                                                nstack=self.nstack)
            with tf.name_scope('test'):
                for i in range(self.partnum):
                    tf.summary.scalar(self.joints[i], self.acc[i], collections=['test'])

        with tf.device(self.cpu):

            with tf.name_scope('training'):
                # print(type(self.loss))
                #tf.summary.scalar('loss_0', self.loss[0], collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])

            # with tf.name_scope('summary'):
            #     for i in range(self.nstack):
            #         tf.summary.scalar("stack%d"%i, self.stack_loss[i], collections=['train'])
            #     for j in range(self.partnum):
            #         tf.summary.scalar(self.joints[i], self.part_loss[j], collections=['train'])
            with tf.name_scope('MAE'):
                tf.summary.scalar("MAE", self.mae, collections=['test'])

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

    def training_init(self, nEpochs=10, valStep=3000,showStep=10):
        with tf.name_scope('Session'):
            with tf.device(self.gpu[0]):
                self._init_weight()
                self.saver = tf.train.Saver()
                if self.resume:
                    print("resume from"+self.resume)
                    self.saver.restore(self.Session, self.resume)
                self.train(nEpochs, valStep,showStep)

    def train(self, nEpochs=10, valStep = 3000,showStep=10 ):
        #best_val = open("./best_val.txt", "w")
        best_model_dir = ""
        best_val = 99999
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

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.Session)
        self.Session.run(init)
        last_lr = self.learn_r
        hm_decay = 1
        if self.validIter < self.val_batch_num:
            val_batch_num = self.validIter
        else:
            val_batch_num = self.val_batch_num
        for epoch in range(nEpochs):
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
                    _,lo ,sta_lo, part_lo,summary,last_lr= self.Session.run\
                        ([self.apply_gradient_op,self.loss,self.stack_loss,self.part_loss,self.train_merged,self.lr],
                         feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})

                    self.train_writer.add_summary(summary, epoch * n_step_epoch + n_batch)
                    self.train_writer.flush()

                else:
                    _, lo , last_lr = self.Session.run([self.apply_gradient_op, self.loss,self.lr],
                                             feed_dict={self.last_learning_rate : last_lr, self.h_decay:hm_decay})
                loss += lo[0]
                hm_decay = 1.
                avg_cost += lo[0] / n_step_epoch
                if (n_batch+1) % valStep == 0:
                    if self.valid_record:
                        val_begin = time.time()
                        accuracy_array = np.zeros([1, 14])
                        predictions = dict()
                        predictions['image_ids'] = []
                        predictions['annos'] = dict()
                        val_begin_time = time.time()

                        for i in range(val_batch_num):  # self.validIter
                            val_percent = ((i + 1) / val_batch_num) * 100
                            val_num = np.int(20 * val_percent / 100)
                            val_tToEpoch = int((time.time() - val_begin) * (100 - val_percent) / (val_percent))

                            accuracy_pred, val_out, val_size, val_name,last_lr = self.Session.run(
                                [self.acc, self.valid_output.outputs, self.valid_size, self.valid_name,self.lr],
                                feed_dict={self.last_learning_rate: last_lr, self.h_decay: hm_decay})

                            # print(np.array(accuracy_pred).shape)
                            accuracy_array += np.array(accuracy_pred, dtype=np.float32) / val_batch_num
                            predictions = getjointcoord(val_out, val_size, val_name, predictions)
                            sys.stdout.write(
                                '\r valid {0}>'.format("=" * val_num) + "{0}>".format(" " * (20 - val_num)) + '||' + str(
                                    percent)[
                                                                                                              :4] +
                                '%' + ' -cost: ' + str(accuracy_array)[:6] +
                                ' -timeToEnd: ' + str(val_tToEpoch) + ' sec.')
                            sys.stdout.flush()
                        print("val done in" + str(time.time() - val_begin_time))
                        score = getScore(predictions, anno, return_dict)

                        print("epoch %d, batch %d ,val score = %d" % (epoch, n_batch, score))
                        tmp = self.mae.assign(score)
                        _ = self.Session.run(tmp)
                        now_acc = (np.sum(accuracy_array) / len(accuracy_array))
                        if now_acc < best_val:

                            best_val = now_acc
                            best_model_dir = os.path.join(self.save_dir, self.name + '_' + str(epoch) +
                                                             "_" + str(n_batch) +"_"+(str(now_acc)[:8]))
                            print("get lower loss, save at " + best_model_dir)
                            with tf.name_scope('save'):
                                self.saver.save(self.Session, best_model_dir)
                            hm_decay = 1.
                        # else:
                        #     print("now val loss is not best, restore model from" + best_model_dir)
                        #     self.saver.restore(self.Session, best_model_dir)
                        #     hm_decay = self.human_decay




                        print('--Avg. Accuracy loss =', str(now_acc)[:6], )
                        self.resume['accur'].append(accuracy_array)
                        self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                        valid_summary = self.Session.run([self.valid_merge])

                        self.valid_writer.add_summary(valid_summary[0], epoch * n_step_epoch + n_batch)
                        self.valid_writer.flush()


            epochfinishTime = time.time()
            print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                ((epochfinishTime - epochstartTime) / n_step_epoch))[:4] + ' sec.')

            self.resume['loss'].append(loss)

            #####valid


        coord.request_stop()
        coord.join(threads)
        self.Session.close()
        print('Training Done')
