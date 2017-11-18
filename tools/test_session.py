import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
import numpy as np
from dataGenerator.datagen import DataGenerator
from train import process_config
from train import process_network
from tools.draw_point_from_test import draw_pic
from eval.ht2coord import getjointcoord
from tools.lr import get_lr
from tools.keypoint_eval import getScore
from tools.keypoint_eval import load_annotations
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
params = process_config('./config/config_mini.cfg')
network_params = process_network("./config/hourglass_mini.cfg")
#network_params = process_network("./config/hgattention.cfg")

show_step = params["show_step"]
test_data = DataGenerator(imgdir=params['train_img_path'], nstack= network_params['nstack'],label_dir=params['label_dir'],
                           out_record=params['train_record'],num_txt=params['train_num_txt'],
                           batch_size=params['batch_size'], name="train_mini", is_aug=False,isvalid=True,scale=
                           params['scale'], refine_num = 10000)
img, hm,sz,nm= test_data.getData()
init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
img_idr = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"
predictions = dict()
predictions['image_ids'] = []
predictions['annos'] = dict()
return_dict = dict()
return_dict['error'] = None
return_dict['warning'] = []
return_dict['score'] = None
anno = load_annotations(params["label_dir"], return_dict)

label_tmp = pd.read_json(params["label_dir"])


with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(init)
    for i in range(1):
        test_img, test_out, val_size, val_name, = sess.run(
            [img, hm, sz, nm], )
        draw_pic(heatmap=test_out, image_size=val_size, image_name=val_name, img_dir=img_idr, ori=test_img, pd = label_tmp)
        # predictions = getjointcoord(test_out, val_size, val_name, predictions)
        # score = getScore(predictions, anno, return_dict)
        # print(score)
    # try:
    #     sess.run(init)
    #     step = 0
    #     im, ht = sess.run([img, hm,])
    #     draw_pic(heatmap=test_out, image_size=val_size, image_name=val_name, img_dir=img_idr, ori=test_img)
    #     print(np.unique(im))
    # except tf.errors.OutOfRangeError:
    #     print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    # finally:
    #     # When done, ask the threads to stop.
    #     coord.request_stop()
    #
    #     # Wait for threads to finish.
    #     coord.join(threads)
    #     sess.close()
    #
    #
