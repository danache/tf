import numpy as np
# import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import tensorflow as tf
import pandas as pd
import tensorlayer as tl
from tools.img_tf import *

class DataGenerator():
    def __init__(self, imgdir=None, label_dir=None, out_record=None, num_txt="", nstack=4, resize=256, scale=0.25,
                 flipping=False,
                 color_jitting=30, rotate=30, batch_size=32, name="", is_aug=True, isvalid=False, refine_num=None):
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
        if os.path.isdir(out_record):
            print(out_record)
            print("record file exist!!")
            self.record_path = out_record
            txt = open(num_txt, "r")

            for line in txt.readlines():
                self.number = int(line.strip())

        else:
            print(self.name + "record file not exist!  creating !!!")
            os.mkdir(out_record)
            self.generageRecord(imgdir, label_dir, out_record, extension=self.scale, resize=256)
            self.record_path = out_record

    def getData(self):
        return self.read_and_decode(filepath =self.record_path, flipping=self.flipping,
                                    color_jitting=self.color_jitting, rotate=self.rotate, batch_size=self.batch_size,
                                    isvalid=self.isvalid)

    def getN(self):
        return self.number

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # def augment(self,):
    #     #包括resize to size, scaling ,fliping, color jitting, rotate,

    def generageRecord(self, imgdir, label_tmp, out_record, extension=0.3, resize=256):

        self.number = 0
        label_tmp = pd.read_json(label_tmp)
        for index, row in label_tmp.iterrows():
            anno = row["human_annotations"]
            #         if(len(anno.keys())  == 1):
            #             continue
            img_path = os.path.join(imgdir, row["image_id"] + ".jpg")

            img = scm.imread(img_path)

            w, h = img.shape[1], img.shape[0]
            keypoint = row["keypoint_annotations"]
            i = 0
            fileName = ("%.6d.tfrecords" % (index))
            writer = tf.python_io.TFRecordWriter(os.path.join(out_record,fileName))

            for key in anno.keys():
                i += 1
                if (anno[key][0] >= anno[key][2] or anno[key][1] >= anno[key][3]):
                    print(img_path)
                    continue

                x1, y1, x2, y2 = anno[key][0], anno[key][1], anno[key][2], anno[key][3]

                board_w = x2 - x1
                board_h = y2 - y1
                center = np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
                ankle = keypoint[key].copy()

                new_img = img.astype(np.uint8)

                feature = {
                    'label': self._bytes_feature(tf.compat.as_bytes(np.array(ankle).astype(np.int32).tostring())),
                    'img_raw': self._bytes_feature(tf.compat.as_bytes(new_img.tostring())),
                    'center': self._bytes_feature(tf.compat.as_bytes(center.astype(np.float32).tostring())),
                    'h': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                    'w': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                    'bh': tf.train.Feature(int64_list=tf.train.Int64List(value=[board_h])),
                    'bw': tf.train.Feature(int64_list=tf.train.Int64List(value=[board_w])),
                    'img_name': self._bytes_feature(tf.compat.as_bytes(row["image_id"])),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                self.number += 1
            writer.close()
            if index % 100 == 0:
                print("creating -- %d" % (index))
            if self.refine_num:
                if index > self.refine_num:
                    break

        txt = open(self.num_txt, "w")
        txt.write(str(self.number))
        txt.close()
        return None

    def _makeGaussian(self, height, width, sigma=3., center=None, flag=True):
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

    def planB(self, height, width):
        return tf.zeros((height, width))

    def generateHeatMap(self, height, width, joints, num_joints, maxlenght):

        hm = []
        coord = []
        for i in range(int(num_joints)):
            tmp = (tf.sqrt(maxlenght) * maxlenght * 10 / 4096.) + 2
            s = tf.cast(tmp, tf.int32)
            x = tf.cast(joints[i * 3], tf.float64)
            y = tf.cast(joints[i * 3 + 1], tf.float64)

            ht = tf.cond(
                (tf.equal(joints[i * 3 + 2], 1.)),
                lambda: self._makeGaussian(height, width, s,
                                           center=(tf.cast(x, tf.int32), tf.cast(y, tf.int32))),
                lambda: self.planB(height, width)
            )
            ht = tf.expand_dims(ht, -1)
            hm.append(ht)

        return hm

    def read_and_decode(self, filepath, img_size=256, label_size=14, heatmap_size=64, flipping=False,
                        color_jitting=True, rotate=30, batch_size=32, isvalid=False):

        feature = {'img_raw': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string),
                   'center': tf.FixedLenFeature([], tf.string),
                   'h': tf.FixedLenFeature([], tf.int64),
                   'w': tf.FixedLenFeature([], tf.int64),
                   'bh': tf.FixedLenFeature([], tf.int64),
                   'bw': tf.FixedLenFeature([], tf.int64),

                   'img_name': tf.FixedLenFeature([], tf.string),
                   }
        # Create a list of filenames and pass it to a queue
        file_lst = []
        for root, dirs, files in os.walk(filepath):
            for name in files:
                file_lst.append(os.path.join(root, name))


        filename_queue = tf.train.string_input_producer(file_lst)
        # Define a reader and read the next record

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers



        center = tf.decode_raw(features['center'], tf.float32)
        center = tf.reshape(center, [2, ])

        label = tf.decode_raw(features['label'], tf.int32)
        label = tf.reshape(label, [label_size, 3])
        label = tf.cast(label, tf.float32)

        height = tf.cast(features['h'], tf.int32)
        width = tf.cast(features['w'], tf.int32)

        boxh = tf.cast(features['bh'], tf.int32)

        boxw = tf.cast(features['bw'], tf.int32)

        img = tf.decode_raw(features['img_raw'], tf.uint8)

        # Cast label data into int32
        # label = tf.cast(features['label'],tf.float32)
        # Reshape image data into the original shape

        img_name = features['img_name']
        res_256 = tf.constant([img_size, img_size], dtype=tf.float32)
        res_64 = tf.constant([heatmap_size, heatmap_size], dtype=tf.float32)

        # return center,boxh,boxw




        scale = tf.stack([boxh, boxw], axis=0)
        scale = tf.cast(scale, tf.float32)
        img = tf.reshape(img, [height, width, 3])

        crop_img = crop(img, height, width, center, scale, res_256, )
        crop_img.set_shape([img_size, img_size, 3])

        coord = transformPreds(coords=label[:, 0:2], center=center,
                               scale=scale, res=res_64)
        coord = tf.squeeze(coord)
        label_exp = tf.expand_dims(label[:, -1], -1)
        coord = tf.squeeze(tf.reshape(tf.concat([coord, label_exp], axis=1), [-1, 1]))
        heatmap = self.generateHeatMap(heatmap_size, heatmap_size, coord, label_size, heatmap_size * 1.)
        repeat = []
        for i in range(len(heatmap)):
            heatmap[i] = tf.squeeze(heatmap[i])
        heatmap = tf.stack(heatmap, axis=-1)
        # for i in range(self.nstack):
        #     repeat.append(heatmap)
        # heatmap = tf.stack(repeat, axis=0)
        crop_img = tf.divide(crop_img,255)
        if batch_size:

            min_after_dequeue = 10
            capacity = min_after_dequeue + 4 * batch_size
            return tf.train.shuffle_batch([crop_img,heatmap, center, scale, img_name],
                                          batch_size=batch_size,
                                          num_threads=4,
                                          capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)

        else:
            return img, label
