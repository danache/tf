import time
import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import os
import cv2
#from tools.draw_point_from_test import draw_pic
import pandas as pd
class test_class():
    def __init__(self, model, nstack=4, test_record="",resume="",gpu=[0],
                 partnum=14,test_img_dir = "",test_json=""
                 ):

        self.resume = resume

        self.test_record = test_record
        self.test_json = test_json

        self.gpu = gpu
        self.cpu = '/cpu:0'
        self.model = model
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]

        self.colors = [
            [ 0, 0,255], [0, 255, 0], [255,0,0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                  [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

        self.mae = tf.Variable(0, trainable=False, dtype=tf.float32,)
        self.test_img_dir = test_img_dir


    def generateModel(self):

        # test_data = self.test_record
        # self.train_num = test_data.getN()
        # self.test_img, test_ht, self.test_size, self.test_name = test_data.getData()
        self.test_img = tf.placeholder(tf.float32,shape=[None,256,256,3])
        self.test_out = self.model(self.test_img,reuse=False).outputs

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

    def test_init(self,img_path="",save_dir="",score=False):
        with tf.name_scope('Session'):
            with tf.device("/gpu:1"):
                self._init_weight()
                self.saver = tf.train.Saver()
                self.init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

                self.coord = tf.train.Coordinator()
                self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.Session)
                self.Session.run(self.init)
                if self.resume:
                    print("resume from"+self.resume)
                    self.saver.restore(self.Session, self.resume)


                self.test(img_path,save_dir)

    def test(self,  img_path,save_dir):

        json = pd.read_csv(self.test_json)
        for index, row in json.iterrows():
            name = os.path.splitext(row['name'])[0]

            img_dir = os.path.join(img_path,name+".jpg" )
            if not os.path.exists(img_dir):
                continue
            #print(img_dir)
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            img = cv2.imread(img_dir)
            human = img[y1:y2, x1:x2]
            img_padd = np.zeros([256,256,3])
            board_w = x2 - x1
            board_h = y2 - y1
            resize = 256

            if board_h < board_w:
                newsize = (resize, board_h * resize // board_w)
            else:
                newsize = (board_w * resize // board_h, resize)
            img_reshape = cv2.resize(human, newsize)
            if (img_reshape.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                up = 0
                down = img_reshape.shape[0]
                img_padd[up:down, :, :] = img_reshape
            elif (img_reshape.shape[1] < resize):
                left = 0
                right = img_reshape.shape[1]
                img_padd[:, left:right, :] = img_reshape
            img_padd_tf= np.expand_dims(img_padd,0)
            hg = self.Session.run(self.test_out,feed_dict={self.test_img:img_padd_tf})
            htmap = hg[0, 3]
            res = np.ones(shape=(14, 3)) * -1

            for joint in range(14):
                idx = np.unravel_index(htmap[:, :, joint].argmax(), (64, 64))
                tmp_idx = np.asarray(idx) * 4
                res[joint][0] = tmp_idx[1]
                res[joint][1] = tmp_idx[0]

            for i in range(14):
                cv2.circle(img_padd, (int(res[i][0]), int(res[i][1])), 5, self.colors[i], -1)
            cv2.imwrite(os.path.join(save_dir, name + ".jpg"), img_padd)



        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Training Done')
