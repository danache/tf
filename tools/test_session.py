import sys
sys.path.append(sys.path[0])
del sys.path[0]
import tensorflow as tf
import numpy as np
import configparser
import time
from dataGenerator.datagen import DataGenerator
from train import process_config


params = process_config('../config.cfg')
train_data = DataGenerator(imgdir=params['train_img_path'], label_dir=params['label_dir'],
                               out_record=params['train_record'],num_txt=params['train_num_txt'],
                               batch_size=params['batch_size'], name="train", is_aug=False,isvalid=False)
img, hm,= train_data.getData()
init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
mae = tf.Variable(0, dtype=tf.float32)

with tf.name_scope('MAE'):
    tf.summary.scalar("MAE", mae, collections=['test'])

valid_merge = tf.summary.merge_all('test')
valid_writer = tf.summary.FileWriter("/home/bnrc2/mu/tf/log/test.log")
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        sess.run(init)
        step = 0
        while not coord.should_stop():
            step += 1
            tmp = mae.assign(step)
            _ = sess.run(tmp)
            summary = sess.run(valid_merge)

            valid_writer.add_summary(summary, step)
            # start_time = time.time()
            # im, ht,imgsize,_name = sess.run([img, hm, img_size, name])
            # print(imgsize,_name)
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


