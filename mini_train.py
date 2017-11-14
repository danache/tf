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




model = HourglassModel(train_img_path=train_img_path,train_label_path=label_dir,train_record=train_record,
                       valid_img_path=valid_img_path, valid_label_path=valid_label, valid_record=valid_record,
                       model_dir="/media/bnrc2/_backup/golf/model/",resume="/media/bnrc2/_backup/golf/model/tiny_hourglass_21")
model.generateModel()
model.training_init(nEpochs=200)
#model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
#                    dataset=None)

