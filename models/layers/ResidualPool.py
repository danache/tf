import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d

from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.activations import relu
def convBlock(data,numIN, numOut, name = ""):
    with tf.variable_scope(name):
        bn1 = tl.layers.BatchNormLayer(data, name="bn1",act=tf.nn.relu)
        conv1 = conv_2d(bn1,numOut / 2,filter_size=(1,1), name="conv1")

        bn2 = tl.layers.BatchNormLayer(conv1, name="bn2",act=tf.nn.relu)

        conv2 = conv_2d(bn2, numOut / 2, filter_size=(3, 3),padding='SAME', name="conv2")

        bn3 = tl.layers.BatchNormLayer(conv2, name="bn3",act=tf.nn.relu)

        conv3 = conv_2d(bn3, numOut, filter_size=(1, 1), name="conv3")

        return conv3

def skipLayer(data,numin, numOut,name=""):
    if numin == numOut:
        return data
    else:
        with tf.variable_scope(name):
            return conv_2d(data,numOut,filter_size=(1,1),strides=(1,1),name="conv")

def poolLayer(data, numOut,name="",suffix=""):
    with tf.variable_scope(name):
        bn1 = tl.layers.BatchNormLayer(data, name="bn1",act=tf.nn.relu)

        pool1 = tl.layers.MaxPool2d(bn1, (2,2),strides=(2,2),name='pool1' )


        conv1 = conv_2d(pool1, numOut, filter_size=(3,3), strides=(1,1),padding='SAME',
                                      name='conv1')
        bn2 = tl.layers.BatchNormLayer(conv1,name='bn2',act=tf.nn.relu)


        conv2 =  conv_2d(bn2, numOut, filter_size=(3,3), strides=(1,1),
                         padding='SAME', name='conv2' )
        x = tl.layers.UpSampling2dLayer(conv2,size=[2,2],is_scale=True, method=1,name="upsample")
        return x

def ResidualPool(data,numin, numOut,name=""):

    convb = convBlock(data,numin,numOut, name="%s_conv_block"%(name))
    skip = skipLayer(data, numin,numOut, name="%s_skip_layer"%(name))
    pool = poolLayer(data,numOut, name="%s_pool_layer"%(name))
    x = tl.layers.ElementwiseLayer(layer=[convb,skip,pool],
                                   combine_fn=tf.add, name="%s_add_layer" % (name))
    return x