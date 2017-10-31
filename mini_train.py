from models.hgattention import createModel
import tensorlayer as tl
import numpy as np
import tensorflow as tf
img = tf.placeholder(tf.float32, shape=[1, 256, 256,3],name='x')
label = tf.placeholder(tf.float32, shape=[1, 8,14,64, 64])

y = createModel(img)
ce = tf.reduce_mean(tf.nn.l2_loss(y, label))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())


"""
out = []
for i in range(8):
    out.append(np.random.rand(1,14,64,64))

d = tf.stack(out,axis=1)
with tf.Session() as sess:
    print(sess.run(d).shape)
"""