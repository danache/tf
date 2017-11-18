import time
import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
import numpy as np
import tensorlayer as tl

from tools.draw_point_from_test import draw_pic
class test_class():
    def __init__(self, model, nstack=4, test_record="",resume="",gpu=[0],
                 partnum=14,test_img_dir = ""
                 ):

        self.resume = resume

        self.test_record = test_record
        self.gpu = gpu
        self.cpu = '/cpu:0'
        self.model = model
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]

        self.mae = tf.Variable(0, trainable=False, dtype=tf.float32,)
        self.test_img_dir = test_img_dir


    def generateModel(self):

        test_data = self.test_record
        self.train_num = test_data.getN()
        self.test_img, test_ht, self.test_size, self.test_name = test_data.getData()
        self.test_out = self.model(self.test_img,reuse=False)

        #self.train_output = self.model(train_img)

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.Session.run(tf.global_variables_initializer())

        self.Session.run(tf.local_variables_initializer())
        tl.layers.initialize_global_variables(self.Session)
        print("init done")

    def training_init(self,):
        with tf.name_scope('Session'):
            with tf.device("/gpu:1"):
                self._init_weight()
                self.saver = tf.train.Saver()
                if self.resume:
                    print("resume from"+self.resume)
                    self.saver.restore(self.Session, self.resume)
                self.test()

    def test(self,  ):


        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.Session)
        self.Session.run(init)

        test_num = 2
        img_idr = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"
        for n_batch in range(test_num):#n_step_epoch
            test_img,test_out, val_size, val_name,  = self.Session.run(
                [self.test_img,self.test_out.outputs, self.test_size, self.test_name],)
            print(np.unique(test_img))
            draw_pic(heatmap=test_out,image_size=val_size,image_name=val_name,img_dir = img_idr,ori = test_img)

        coord.request_stop()
        coord.join(threads)
        self.Session.close()
        print('Training Done')
