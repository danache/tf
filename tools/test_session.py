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
print(params)
train_data = DataGenerator(imgdir=params['train_img_path'], label_dir=params['label_dir'],
                               out_record=params['train_record'],
                               batch_size=params['batch_size'], name="train", is_aug=False)
img, hm, img_size, name = train_data.getData()
init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        sess.run(init)
        step = 0
        while not coord.should_stop():
            start_time = time.time()
            im, ht,imgsize,_name = sess.run([img, hm, img_size, name])
            print(imgsize,_name)
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


