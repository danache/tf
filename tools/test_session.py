import tensorflow as tf
import numpy as np
import time
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
            im, ht = sess.run([img, hm])
            print(np.unique(np.array(im)))
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


