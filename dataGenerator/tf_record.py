
# coding: utf-8

# In[1]:

import sys 
sys.version


# In[2]:

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


# In[3]:

imgdir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902"
label_dir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json"
label_to = pd.read_json(label_dir)


# In[99]:

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# In[171]:

def generageRecord(imgdir,label_tmp,out_record,extension=0.3,resize=256):
    writer = tf.python_io.TFRecordWriter(out_record)  
    for index, row in label_tmp.iterrows():
        anno = row["human_annotations"]
#         if(len(anno.keys())  == 1):
#             continue
        img_path = os.path.join(imgdir , row["image_id"]+".jpg")
        
        img = cv2.imread(img_path)
        w, h = img.shape[1], img.shape[0]
        keypoint = row["keypoint_annotations"]
        i = 0
        for key in anno.keys():
            i += 1
            if (anno[key][0] >= anno[key][2] or anno[key][1] >= anno[key][3]):
                print(img_path)
                continue
                
            x1,y1,x2,y2 = anno[key][0],anno[key][1],anno[key][2],anno[key][3]
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
                newsize = (board_w * resize// board_h, resize)
            for j in range(len(ankle)):
                    if j % 3 == 0:
                        ankle[j] =(ankle[j]- x1) / board_w
                    elif j % 3 == 1:
                        ankle[j] =(ankle[j]- y1) /board_h
                    else:
                        ankle[j] =ankle[j] * 1.
                        
            #print(ankle)
                        
            tmp = cv2.resize(human, newsize)
            new_img = np.zeros((resize, resize, 3))
            if(tmp.shape[0] <resize):#高度不够，需要补0。则要对item[6:]中的第二个值进行修改
                
                up = np.int((resize - tmp.shape[0]) * 0.5)
                down = np.int((resize + tmp.shape[0]) * 0.5)
                new_img[up:down,:,:] = tmp
                for j in range(len(ankle)):
                    if j % 3 == 1:
                        ankle[j] = (tmp.shape[0] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[0]))/ resize
            elif(tmp.shape[1] < resize):
                left = np.int((resize - tmp.shape[1]) * 0.5)
                right = np.int((resize + tmp.shape[1]) * 0.5)
                new_img[:, left:right ,:] = tmp
                for j in range(len(ankle)):
                    if j % 3 == 0:
                        ankle[j] = (tmp.shape[1] * ankle[j] * 1. + 0.5 * (resize - tmp.shape[1]))/ resize
            #print(ankle)
#             for j in range(14):
#                 coord = ankle[j * 3: j * 3 + 2]
#                 if coord[-1] != 2:
#                     human = cv2.circle(new_img,(int(coord[0] * resize),int(coord[1]* resize)),10,(255,0,255),-1)
            new_img = new_img.astype(np.int8)

            feature = {'label': _bytes_feature(tf.compat.as_bytes(np.array(ankle).astype(np.float32).tostring()))
                       ,'img_raw': _bytes_feature(tf.compat.as_bytes( new_img.tostring()))}
    
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())    
        if index > 0:
            break
        writer.close()  
    return None


# In[172]:

generageRecord(imgdir=imgdir,label_tmp=label_to,out_record="./test.tfrecords")


# # 生成heatmap

# In[173]:

def _makeGaussian(height, width, sigma=3., center=None,flag=True):
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
    
    
    dx = tf.pow(tf.subtract(x , x0),2)
    dy = tf.pow(tf.subtract(y , y0),2)
    fenzi = tf.multiply(tf.multiply(tf.add(dx , dy), tf.log(2.)), -4.0)
    fenmu = tf.cast(tf.pow(sigma, 2),tf.float32)
    dv = tf.divide(fenzi,fenmu)
    return tf.exp(dv)
def planB(height, width):
    return tf.zeros((height, width))


# # 使用队列

# In[174]:

def generateHeatMap(height, width, joints,num_joints, maxlenght):

    hm = []
    coord = []
    for i in range(int(num_joints)):
        tmp  = (tf.sqrt(maxlenght) * maxlenght * 10 / 4096.) + 2
        s =  tf.cast(tmp,tf.int32)     
        
        x = tf.cast(joints[i * 3],tf.float64)
        y = tf.cast(joints[i * 3 + 1],tf.float64)
        #print(tf.(joints[i * 3 + 2], 1.))
        ht= tf.cond(
            (tf.equal(joints[i * 3 + 2], 1.)),
            lambda: _makeGaussian(height,width,s,center=(tf.cast(x * 64,tf.int32),tf.cast(y * 64,tf.int32))),
            lambda: planB(height, width)
            )
        hm.append(ht)
        
    return hm
                


# In[188]:

def read_and_decode(filename,img_size=256,label_size=42,heatmap_size=64,scale=0.25,flipping=False,
                    color_jitting=True,rotate=30):

    feature = {'img_raw': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    # Define a reader and read the next record
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    
    img = tf.decode_raw(features['img_raw'], tf.int8)
    
    # Cast label data into int32
    #label = tf.cast(features['label'],tf.float32)
    # Reshape image data into the original shape
    img = tf.reshape(img, (256, 256, 3))
    
    
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [label_size,])
    
    """
    Data augmention
    """
    ###random_scale
    if scale:
        img_scale_size =  int(random.uniform(1-scale,1+scale) *  img_size)
        img = tf.image.resize_images(img,(img_scale_size,img_scale_size),method=tf.image.ResizeMethod.BICUBIC)

    heatmap= generateHeatMap(heatmap_size,heatmap_size, label,label_size / 3,heatmap_size*1.)
    
    ###rotate
    if rotate:
        rotate_angle = random.randint(-rotate,rotate)
        img = tf.contrib.image.rotate(img,angles=rotate_angle)
        for i in range(len(heatmap)):
                heatmap[i] = tf.contrib.image.rotate(heatmap[i],angles=rotate_angle)
    ###flip
    if flipping:
        if(random.random() > 0.5):
            img = tf.image.flip_left_right(img)
            for i in range(len(heatmap)):
                heatmap[i] = tf.image.flip_left_right(heatmap[i])
    if color_jitting:
        ###color_jitting
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.3, upper=1.0)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_saturation(img, lower=0.0, upper=2.0)
    

    return img,label,heatmap


# In[189]:

#img, labels = read_and_decode("test.tfrecords")
img ,label,heatmap= read_and_decode("test.tfrecords")
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
print(tf.__version__)
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(init)
    for i in range(1):
        if coord.should_stop():
            break
        i,l,h= sess.run([img,label,heatmap])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12) 
        
        
 
    coord.request_stop()
    coord.join(threads)
    


# In[180]:

im = np.array(im)


# In[197]:

ttt = np.array(h[0])


# In[199]:

np.unique(ttt)


# In[190]:

im = np.squeeze(i)
cv2.imwrite("./ooo.jpg",im)


# In[ ]:



