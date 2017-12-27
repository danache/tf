import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
from dataGenerator.datagen_v2 import DataGenerator
from train import process_config
from train import process_network
from tools.keypoint_eval import load_annotations
from tools.keypoint_eval import getScore
import pandas as pd
import os
from tools.img_tf import *
from eval.ht2coord import getjointcoord
from four_stack.Hourglass import HourglassModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
params = process_config('./config/config_2.cfg')
network_params = process_network("./config/hourglass.cfg")
#network_params = process_network("./config/hgattention.cfg")

show_step = params["show_step"]
test_data = DataGenerator(imgdir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/", nstack= 2,label_dir="/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
                               out_record="/media/bnrc2/_backup/dataset/new_tfrecord/test.tfrecords",num_txt="/media/bnrc2/_backup/dataset/new_tfrecord/test.txt",
                               batch_size=1, name="train_mini", is_aug=False,isvalid=False, refine_num = 10)

img, heatmap, center, scale, img_name= test_data.getData()


init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
img_idr = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"



label_tmp = pd.read_json(params["label_dir"])

resume=params["resume"]

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # model = HourglassModel(nFeat=network_params['nfeats'], nStack=network_params['nstack'],
    #                        nModules=network_params['nmodules'], outputDim=network_params['partnum'])._graph_hourglass
    # output = model(img, reuse=False).outputs
    sess.run(init)
    # saver = tf.train.Saver()
    # saver.restore(sess,resume)

    for i in range(20):
        train_img, train_ht, train_center, train_scale,train_name, = sess.run(
            [img, heatmap, center, scale, img_name], )
        train_out = sess.run(output)
        #
        # predictions = dict()
        # predictions['image_ids'] = []
        # predictions['annos'] = dict()
        # return_dict = dict()
        # return_dict['error'] = None
        # return_dict['warning'] = []
        # return_dict['score'] = None
        # anno = load_annotations(params["label_dir"], return_dict)
        #
        # #draw_pic(heatmap=test_out, image_size=val_size, image_name=val_name, img_dir=img_idr, ori=test_img, pd = label_tmp)
        # predictions = getjointcoord(train_ht, train_size, train_name, predictions)
        # score = getScore(predictions, anno, return_dict)
        # print("gt score = ")
        # print(score)
        out_predictions = dict()
        out_predictions['image_ids'] = []
        out_predictions['annos'] = dict()
        out_return_dict = dict()
        out_return_dict['error'] = None
        out_return_dict['warning'] = []
        out_return_dict['score'] = None
        anno2 = load_annotations(params["label_dir"], out_return_dict)
        out_predictions = getjointcoord(train_out, train_center, train_scale,train_name, out_predictions)
        outscore = getScore(out_predictions, anno2, out_return_dict)
        print("predscore = ")
        print(outscore)
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
