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

train_img_path = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902"
label_dir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json"
train_record = "/media/bnrc2/_backup/ai/mu/train.tfrecords"
valid_img_path = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/"
valid_label = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"
valid_record = "/media/bnrc2/_backup/ai/mu/valid.tfrecords"

model = HourglassModel(train_img_path=train_img_path,train_label_path=label_dir,train_record=train_record,
                       valid_img_path=valid_img_path, valid_label_path=valid_label, valid_record=valid_record,
                       model_dir="/media/bnrc2/_backup/golf/model/",resume="/media/bnrc2/_backup/golf/resume/tiny_hourglass_19")
model.generateModel()
model.training_init(nEpochs=200)
#model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],
#                    dataset=None)

