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
    def __init__(self, imgdir=None, label_dir=None, out_record=None, resize=256,scale=0.25, flipping=False,
                 color_jitting=30,rotate=30):
        if os.path.exists(out_record):
            self.record_path = out_record
        else:
            self.generageRecord(imgdir, label_dir, out_record, extension=0.3, resize=256)
            self.record_path = out_record
        self.resize = resize
        self.scale = scale
        self.flipping = flipping
        self.color_jitting = color_jitting
        self.rorate = rotate


    # def augment(self,):
    #     #包括resize to size, scaling ,fliping, color jitting, rotate,
    def generageRecord(self,imgdir, label_tmp, out_record, extension=0.3, resize=256):
        writer = tf.python_io.TFRecordWriter(out_record)
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
                if (tmp.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                    new_img = np.zeros((resize, resize, 3))
                    up = np.int((resize - tmp.shape[0]) * 0.5)
                    down = np.int((resize + tmp.shape[0]) * 0.5)
                    new_img[up:down, :, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 1:
                            ankle[j] = (tmp.shape[0] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[0])) / resize
                elif (tmp.shape[1] < resize):
                    new_img = np.zeros((resize, resize, 3))
                    left = np.int((resize - tmp.shape[1]) * 0.5)
                    right = np.int((resize + tmp.shape[1]) * 0.5)
                    new_img[:, left:right, :] = tmp
                    for j in range(len(ankle)):
                        if j % 3 == 0:
                            ankle[j] = (tmp.shape[1] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[1])) / resize

                img_raw = new_img.tobytes()
                label_raw = np.array(ankle).tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
            writer.close()
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
            # print(tf.(joints[i * 3 + 2], 1.))
            ht = tf.cond(
                (tf.equal(joints[i * 3 + 2], 1.)),
                lambda: self._makeGaussian(height, width, s, center=(tf.cast(x * 64, tf.int32), tf.cast(y * 64, tf.int32))),
                lambda: self.planB(height, width)
            )
            hm.append(ht)

        return hm

    def read_and_decode(self, img_size=256, label_size=42, heatmap_size=64, scale=0.25, flipping=False,
                        color_jitting=True, rotate=30):

        filename_queue = tf.train.string_input_producer([self.record_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['img_raw'], tf.float64)
        img = tf.reshape(img, [img_size, img_size, 3])

        label = tf.decode_raw(features['label'], tf.float64)
        label = tf.reshape(label, [label_size, ])

        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        """
        Data augmention
        """
        ###random_scale
        if scale:
            img_scale_size = int(random.uniform(1 - scale, 1 + scale) * img_size)
            img = tf.image.resize_images(img, (img_scale_size, img_scale_size), method=tf.image.ResizeMethod.BICUBIC)

        heatmap = self.generateHeatMap(heatmap_size, heatmap_size, label, label_size / 3, heatmap_size * 1.)

        ###rotate
        if rotate:
            img = tf.contrib.image.rotate(img, angles=rotate)
            for i in range(len(heatmap)):
                heatmap[i] = tf.contrib.image.rotate(heatmap[i], angles=rotate)
        ###flip
        if flipping:
            if (random.random() > 0.5):
                img = tf.image.random_flip_left_right(img)
                for i in range(len(heatmap)):
                    heatmap[i] = tf.image.random_flip_left_right(heatmap[i])
        if color_jitting:
            ###color_jitting
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_saturation(img, lower=0.0, upper=2.0)


            #     return img,label
        return img, heatmap