import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import tensorflow as tf
import pandas as pd
import tensorlayer as tl
"""
自定义数据生成器，包括生成tfrecord,数据增强，迭代器等。
对于不同的数据集,根据图片数据集不同改写。
"""
class DataGenerator():
    def __init__(self, imgdir=None, label_dir=None, out_record=None, num_txt="",nstack = 4,resize=256,scale=0.25, flipping=False,
                 color_jitting=30,rotate=30,batch_size=32,name="",is_aug=True, isvalid=False,refine_num=None):
        self.nstack = nstack
        if is_aug:

            self.flipping = flipping
            self.color_jitting = color_jitting
            self.rotate = rotate
        else:

            self.flipping = False
            self.color_jitting = False
            self.rotate = False
        self.num_txt = num_txt
        self.scale = scale
        self.isvalid = isvalid
        self.resize = resize
        self.batch_size = batch_size
        self.name = name
        self.refine_num = refine_num
        if self.refine_num:
            print("max num is " + str(refine_num))
        if os.path.exists(out_record):
            print(out_record)
            print("record file exist!!")
            self.record_path = out_record
            txt = open(num_txt,"r")

            for line in txt.readlines():
                self.number = int(line.strip())

        else:
            print(self.name + "record file not exist!  creating !!!")
            self.generageRecord(imgdir, label_dir, out_record, extension=self.scale, resize=256)
            self.record_path = out_record

    def getData(self):
        return self.read_and_decode(filename=self.record_path,img_size=self.resize,flipping=self.flipping,scale=self.scale,
                                    color_jitting=self.color_jitting,rotate=self.rotate, batch_size=self.batch_size,
                                    isvalid=self.isvalid)
    def getN(self):
        return self.number

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    # def augment(self,):
    #     #包括resize to size, scaling ,fliping, color jitting, rotate,

    def generageRecord(self,imgdir, label_tmp, out_record, extension=0.3, resize=256):
        writer = tf.python_io.TFRecordWriter(out_record)
        self.number = 0
        label_tmp = pd.read_json(label_tmp)
        for index, row in label_tmp.iterrows():
            anno = row["human_annotations"]
            #         if(len(anno.keys())  == 1):
            #             continue
            img_path = os.path.join(imgdir, row["image_id"] + ".jpg")

            img = cv2.imread(img_path)

            w, h = img.shape[1], img.shape[0]
            keypoint = row["keypoint_annotations"]
            i = 0
            for key in anno.keys():
                i += 1
                if (anno[key][0] >= anno[key][2] or anno[key][1] >= anno[key][3]):
                    print(img_path)
                    continue

                x1, y1, x2, y2 = anno[key][0], anno[key][1], anno[key][2], anno[key][3]
                board_w = x2 - x1
                board_h = y2 - y1
                x1 = 0 if x1 - int(board_w * extension * 0.5) < 0 else x1 - int(board_w * extension * 0.5)
                x2 = w if x2 + int(board_w * extension * 0.5) > w else x2 + int(board_w * extension * 0.5)
                y1 = 0 if y1 - int(board_h * extension * 0.5) < 0 else y1 - int(board_h * extension * 0.5)
                y2 = h if y2 + int(board_h * extension * 0.5) > h else y2 + int(board_h * extension * 0.5)
                board_w = x2 - x1
                board_h = y2 - y1
                human = img[y1:y2, x1:x2]
                ankle = keypoint[key].copy()
                #             print(x1,y1,x2,y2)
                #             print(board_w,board_h)
                #             print(ankle)


                if board_h < board_w:
                    newsize = (resize, board_h * resize // board_w)
                else:
                    newsize = (board_w * resize // board_h, resize)
                for j in range(len(ankle)):
                    if j % 3 == 0:
                        ankle[j] = (ankle[j] - x1) / board_w
                    elif j % 3 == 1:
                        ankle[j] = (ankle[j] - y1) / board_h
                    else:
                        ankle[j] = ankle[j] * 1.

                # print(ankle)

                tmp = cv2.resize(human, newsize)
                new_img = np.zeros((resize, resize, 3))

                """
                中间补0,舍弃，以后用stn变换取代
                if (tmp.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                    up = np.int((resize - tmp.shape[0]) * 0.5)
                    down = np.int((resize + tmp.shape[0]) * 0.5)
                    new_img[up:down, :, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 1:
                            ankle[j] = (tmp.shape[0] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[0])) * 1./ resize
                elif (tmp.shape[1] < resize):
                    left = np.int((resize - tmp.shape[1]) * 0.5)
                    right = np.int((resize + tmp.shape[1]) * 0.5)
                    new_img[:, left:right, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 0:
                            ankle[j] = (tmp.shape[1] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[1])) * 1./ resize
                            # print(ankle)
                            #             for j in range(14):
                            #                 coord = ankle[j * 3: j * 3 + 2]
                            #                 if coord[-1] != 2:
                            #                     human = cv2.circle(new_img,(int(coord[0] * resize),int(coord[1]* resize)),10,(255,0,255),-1)
                """
                ####放在左上角
                if (tmp.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                    up = 0
                    down = tmp.shape[0]
                    new_img[up:down, :, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 1:
                            ankle[j] = (tmp.shape[0] * ankle[j] * 1.) * 1. / resize
                elif (tmp.shape[1] < resize):
                    left = 0
                    right = tmp.shape[1]
                    new_img[:, left:right, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 0:
                            ankle[j] = (tmp.shape[1] * ankle[j] * 1.) * 1. / resize

                new_img = new_img.astype(np.uint8)
                img_size = [ w,h,x1, y1, board_w,board_h,newsize[0], newsize[1]]
                feature = {
                    'label': self._bytes_feature(tf.compat.as_bytes(np.array(ankle).astype(np.float64).tostring())),
                    'img_raw': self._bytes_feature(tf.compat.as_bytes(new_img.tostring())),
                    'img_size': self._bytes_feature(tf.compat.as_bytes(np.array(img_size).astype(np.float64).tostring())),
                    'img_name':self._bytes_feature(tf.compat.as_bytes(row["image_id"]))
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                self.number += 1
            if index % 100 == 0:
                print("creating -- %d" % (index))
            if self.refine_num :
                if index > self.refine_num:
                    break
        writer.close()
        txt = open(self.num_txt,"w")
        txt.write(str(self.number))
        txt.close()
        return None

    def _makeGaussian(self,height, width, sigma=3., center=None, flag=True):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = tf.range(0., width, 1.)
        y = tf.range(0., height, 1.)[:, tf.newaxis]
        if center is None:

            x0 = width // 2
            y0 = height // 2
        else:

            x0 = center[0]
            y0 = center[1]

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x0 = tf.cast(x0, tf.float32)
        y0 = tf.cast(y0, tf.float32)

        dx = tf.pow(tf.subtract(x, x0), 2)
        dy = tf.pow(tf.subtract(y, y0), 2)
        fenzi = tf.multiply(tf.multiply(tf.add(dx, dy), tf.log(2.)), -4.0)
        fenmu = tf.cast(tf.pow(sigma, 2), tf.float32)
        dv = tf.divide(fenzi, fenmu)
        return tf.exp(dv)

    def planB(self,height, width):
        return tf.zeros((height, width))

    def generateHeatMap(self,height, width, joints, num_joints, maxlenght):

        hm = []
        coord = []
        for i in range(int(num_joints)):
            tmp = (tf.sqrt(maxlenght) * maxlenght * 10 / 4096.) + 2
            s = tf.cast(tmp, tf.int32)
            x = tf.cast(joints[i * 3], tf.float64)
            y = tf.cast(joints[i * 3 + 1], tf.float64)
            ht = tf.cond(
                (tf.equal(joints[i * 3 + 2], 1.)),
                lambda: self._makeGaussian(height, width, s, center=(tf.cast(x * 64, tf.int32), tf.cast(y * 64, tf.int32))),
                lambda: self.planB(height, width)
            )
            ht = tf.expand_dims(ht, -1)
            hm.append(ht)

        return hm

    def read_and_decode(self,filename, img_size=256, label_size=42, heatmap_size=64, scale=0.25, flipping=False,
                        color_jitting=True, rotate=30,batch_size=32,isvalid=False):

        feature = {'img_raw': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string),
                   'img_size': tf.FixedLenFeature([], tf.string),
                   'img_name': tf.FixedLenFeature([], tf.string),
                   }
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([filename])
        # Define a reader and read the next record

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers

        img = tf.decode_raw(features['img_raw'], tf.uint8)

        # Cast label data into int32
        # label = tf.cast(features['label'],tf.float32)
        # Reshape image data into the original shape
        img = tf.reshape(img, (img_size, img_size, 3))

        label = tf.decode_raw(features['label'], tf.float64)
        label = tf.reshape(label, [label_size, ])

        img_size = tf.decode_raw(features['img_size'], tf.float64)
        img_size = tf.reshape(img_size, [8,])
        img_name = features['img_name']
        """
        Data augmention
        """
        ###random_scale

        heatmap = self.generateHeatMap(heatmap_size, heatmap_size, label, label_size / 3, heatmap_size * 1.)
        repeat = []


        ###rotate
        if self.rotate:
            print("rotate")
            rotate_angle = random.uniform(-rotate*1./360, rotate*1./360) * np.pi
            #ex_angle = np.pi / 8
            #print(rotate_angle)
            img = tf.contrib.image.rotate(img, angles=rotate_angle)
            for i in range(len(heatmap)):
                heatmap[i] = tf.contrib.image.rotate(heatmap[i], angles=rotate_angle)
        ###flip
        if self.flipping:
            print("flipping")
            if (random.random() > 0.5):
                img = tf.image.flip_left_right(img)
                for i in range(len(heatmap)):
                    heatmap[i] = tf.image.flip_left_right(heatmap[i])
        if self.color_jitting:
            print("color jitting")
            ###color_jitting
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_saturation(img, lower=0.0, upper=2.0)
        for i in range(len(heatmap)):
            heatmap[i] = tf.squeeze(heatmap[i])
        heatmap = tf.stack(heatmap,axis=-1)
        for i in range(self.nstack):
            repeat.append(heatmap)
        heatmap = tf.stack(repeat, axis=0)

        img_mini = tf.expand_dims(img, 0)
        img_mini = tf.image.resize_bilinear(img_mini,(heatmap_size,heatmap_size))
        img_mini = tf.squeeze(img_mini)
        img = tf.cast(img, tf.float32)
        img_mini = tf.cast(img_mini, tf.float32)
        # img = tf.divide(img, 255)

        if batch_size:
            if isvalid:
                min_after_dequeue = 10
                capacity = min_after_dequeue + 4 * batch_size
                image, ht,size,name = tf.train.shuffle_batch([img, heatmap,img_size,img_name],
                                                      batch_size=batch_size,
                                                      num_threads=4,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

                return image, ht,size,name
            else:
                min_after_dequeue = 10
                capacity = min_after_dequeue + 4 * batch_size
                image, img_min, ht,size,name = tf.train.shuffle_batch([img, img_mini,heatmap,img_size,img_name],
                                                               batch_size=batch_size,
                                                               num_threads=4,
                                                               capacity=capacity,
                                                               min_after_dequeue=min_after_dequeue)

                return image, img_min,ht,size,name
        else:
            return img,label


