import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d

def convBlock(data,numIN, numOut, name = "",reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        bn1 = tl.layers.BatchNormLayer(data, name="bn1",act=tf.nn.relu)
        conv1 = conv_2d(bn1,numOut / 2,filter_size=(1,1), name="conv1")

        bn2 = tl.layers.BatchNormLayer(conv1, name="bn2",act=tf.nn.relu)

        conv2 = conv_2d(bn2, numOut / 2, filter_size=(3, 3),padding='SAME', name="conv2")

        bn3 = tl.layers.BatchNormLayer(conv2, name="bn3",act=tf.nn.relu)

        conv3 = conv_2d(bn3, numOut, filter_size=(1, 1), name="conv3")

        return conv3

def skipLayer(data,numin, numOut,name="",reuse=False):
    if numin == numOut:
        return data
    else:
        with tf.variable_scope(name,reuse=reuse):
            return conv_2d(data,numOut,filter_size=(1,1),strides=(1,1),name="conv")

def Residual(data,numin, numOut,name,reuse=False):

    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    with tf.variable_scope(name, reuse=reuse):
        convb = convBlock(data, numin,numOut,name="%s_convBlock" %(name))
        skiplayer = skipLayer(data, numin,numOut, name="%s_skipLayer"%(name))
        x = tl.layers.ElementwiseLayer(layer=[convb, skiplayer],
        combine_fn = tf.add, name="%s_add_layer" % (name))
        #x = tf.add_n([convb, skiplayer],name="%s_add_layer"%(name))
        return x