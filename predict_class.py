import time
import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import os
import cv2
import scipy.misc as scm
#from tools.draw_point_from_test import draw_pic
import pandas as pd

def EnsureDir(dirs):
    if os.path.isdir(dirs):
        return
    else:
        os.mkdir(dirs)

class test_class():
    def __init__(self, model, nstack=4, test_record="",resume="",gpu=[0],
                 partnum=14,dategen=None,save_dir=""
                 ):

        self.resume = resume

        self.test_record = test_record

        self.save_dir = save_dir
        self.gpu = gpu
        self.cpu = '/cpu:0'
        self.model = model
        self.partnum=partnum
        self.joints = ["rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist", "rhip","rknee","rankle",
                       "lhip","lknee","lankle","head","neck"]
        self.datagen = dategen
        self.colors = [
            [ 0, 0,255], [0, 255, 0], [255,0,0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                  [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.line = [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[6,14],[9,14],[0,13],[3,13]]
        self.mae = tf.Variable(0, trainable=False, dtype=tf.float32,)


    def generateModel(self):

        # test_data = self.test_record
        # self.train_num = test_data.getN()
        # self.test_img, test_ht, self.test_size, self.test_name = test_data.getData()
        with tf.variable_scope(tf.get_variable_scope()) as vscope:

            for i in range(len(self.gpu)):
                print("/gpu:%d" % self.gpu[i])
                with tf.device(("/gpu:%d" % self.gpu[i])):
                    with tf.name_scope('gpu_%d' % (self.gpu[i])) as scope:
                        self.train_img = tf.placeholder(shape=[1,256,256,3],dtype=tf.float32)
                        self.test_out= self.model.build(self.train_img)

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

    def test_init(self):
        with tf.name_scope('Session'):
            with tf.device("/gpu:0"):
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


                #self.test(img_path,save_dir)

    def test(self, save_dir):

        json = pd.read_csv(self.test_json)
        # for index, row in json.iterrows():


        img_dir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/c681fcce2fab08692d11100fd8195353cf27a631.jpg"

        #print(img_dir)
        x1, y1, x2, y2 = 333, 98, 511, 667
        img = cv2.imread(img_dir)
        human = img[y1:y2, x1:x2]

        img_reshape = cv2.resize(human, (256,256))

        img_mean =(img_reshape.astype(np.float64)-  np.array([[[102.9801,115.9465,122.7717]]])) / 255

        img_input = img_mean.reshape(1,256,256,3)


        hg = self.Session.run(self.test_out,feed_dict={self.test_img:img_input})

        htmap = hg[-1]

        res = np.ones(shape=(14, 3)) * -1


        for joint in range(14):
            idx = np.unravel_index(htmap[ :, :, joint].argmax(), (64, 64))
            print(idx)
            tmp_idx = np.asarray(idx) * 4

            res[joint][0] = tmp_idx[1]
            res[joint][1] = tmp_idx[0]


        for i in range(14):
            cv2.circle(img_reshape, (int(res[i][0]), int(res[i][1])), 5, self.colors[i], -1)
        cv2.imwrite(os.path.join(save_dir, "abcdf.jpg"), img_reshape)



        self.coord.request_stop()
        self.coord.join(self.threads)
        self.Session.close()
        print('Test Done')

    def pred(self):
        generator = self.datagen.get_batch_generator()
        hm = {}
        coordssss = {}
        while True:
            try:
                train_img, img, center, scale ,name= next(generator)

                hg = self.Session.run([self.test_out],feed_dict={self.train_img:train_img})
                hm[name]= hg[0][-1]
                joint = self.datagen.recoverFromHm( hg[0][-1], center, scale).astype(np.int32)

                hip = np.average(np.stack([joint[6], joint[9]], axis=0), axis=0)
                hip = np.transpose(np.expand_dims(hip, -1))
                coord = np.concatenate([joint, hip], axis=0)
                coord = coord.astype(np.int32)
                coordssss[name] = coord
                for index in range(14):
                    cv2.circle(img, (int(joint[index][0]), int(joint[index][1])), 5, self.colors[index], -1)
                for j in range(len(self.line)):
                    cv2.line(img, (coord[self.line[j][0]][0], coord[self.line[j][0]][1]),
                             (coord[self.line[j][1]][0], coord[self.line[j][1]][1]), (0,255,0), 3)
                img_path = os.path.join(self.save_dir, name)
                folder = "/"+os.path.join(*(img_path.split("/")[:-1]))
                EnsureDir(folder)
                scm.imsave(os.path.join(self.save_dir, name), img)
            except Exception as e:
                np.save(os.path.join(self.save_dir, "byd_d_hm.npy"), np.array(hm))
                np.save(os.path.join(self.save_dir, "byd_d_coord.npy"), np.array(coordssss))

                print(e)
                return


