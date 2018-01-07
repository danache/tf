from four_stack.Hourglass import HourglassModel
import tensorlayer as tl
import tensorflow as tf
import numpy as np
a = tf.placeholder(shape=[1,256,256,3],dtype=tf.float32)

hg = HourglassModel(nFeats=256, nStack=2,
                           nModules=2,outputDim=14)._graph_hourglass

output = hg(a)[-1]

lr = tf.Variable(0.,trainable=False)

sess = tf.Session()
train_writer = tf.summary.FileWriter("./log/", sess.graph)
with tf.name_scope('training'):
    # print(type(self.loss))
    # tf.summary.scalar('loss_0', self.loss[0], collections=['train'])
    tf.summary.scalar('learning_rate', lr, collections=['train'])


train_merged = tf.summary.merge_all('train')
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

sess.run(init)

b ,c= sess.run([output,train_merged],feed_dict={a:np.random.random_sample([1,256,256,3])})
# print(b)
train_writer.add_summary(c, 0)
train_writer.flush()