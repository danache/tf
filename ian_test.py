# coding:utf8
import tensorflow as tf
from four_stack.ian_hourglass import hourglassnet
import tensorflow.contrib.slim as slim


def main():
    # with tf.variable_scope(name_or_scope='foo'):
    #     with tf.variable_scope(name_or_scope='f'):
    #         with tf.variable_scope(name_or_scope='sd'):
    #             # v1 = tf.Variable([2], name='v1')
    #             v1 = tf.get_variable(name='v1', shape=[3, 4])
    #             print(v1)


    hg = hourglassnet()
    out = hg.build()
    print('out--------:', out)
    # feat = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hourglass')
    # for i, var in enumerate(feat):
    #     print(var)

    # for var in slim.get_model_variables(scope='hourglass'):
    #     print(var)


if __name__ == '__main__':
    main()
