from models.hgattention import createModel
import tensorlayer as tl
import numpy as np
import tensorflow as tf
from four_stack.Hourglass import HourglassModel
# img = tf.placeholder(tf.float32, shape=[1, 256, 256,3],name='x')
# label = tf.placeholder(tf.float32, shape=[1, 8,14,64, 64])
#
# y = createModel(img)
# ce = tf.reduce_mean(tf.nn.l2_loss(y, label))
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())


model = HourglassModel()
model.generate_model()
#model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
#                    dataset=None)

