import tensorlayer as tl;
import tensorflow as tf
from tensorlayer.layers import Conv2d


def conv_2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None, padding='SAME',
             name='conv2d'):
    w_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    b_init = tf.constant_initializer(value=0.0)
    return Conv2d(net=net,n_filter=n_filter,filter_size=filter_size,strides=strides,act=act,padding=padding,
                  W_init=w_init,b_init=b_init,name=name)