import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from tflearn.activations import sigmoid
import numpy as np
import opt

def replicate(input, numIn, dim, name):
    with tf.variable_scope(name) as scope:
        repeat = []
        for i in range(numIn):
            repeat.append(input)
        return tl.layers.ConcatLayer(repeat, dim)


def AttentionIter(data, numin,lrnSize, iterSize,name=""):
    lsigmoid = lambda x: tf.nn.sigmoid(x)
    with tf.variable_scope(name) as scope:
        U = conv_2d(data, 1, filter_size=(3, 3), padding='SAME',name="conv1")

        C = []
        Q = []
        for i in range(iterSize):
            if i == 0:
                with tf.variable_scope('spConv', reuse=False):
                    tl.layers.set_name_reuse(False)
                    conv = conv_2d(U, 1,filter_size=(lrnSize,lrnSize),  strides=(1,1),padding='SAME',
                                   name='sp_conv')
            else :
                with tf.variable_scope('spConv', reuse=True):
                    tl.layers.set_name_reuse(True)
                    conv = conv_2d(Q[i - 1], 1,filter_size=(lrnSize,lrnSize),  strides=(1,1),padding='SAME',
                                   name='sp_conv')
            C.append(conv)
            Q_tmp = tl.layers.ElementwiseLayer(layer=[C[i], U],
                                   combine_fn=tf.add, name="%s_add_layer" % (name))
            Q_tmp.outputs = lsigmoid(Q_tmp.outputs)
            Q.append(Q_tmp)

        replicat = replicate(Q[iterSize - 1], numin, -1,name='_replicate')  # ******Q[itersize]-->Q[itersize-1]  2-->3
        pheat = tl.layers.ElementwiseLayer(layer=[data, replicat],
                                   combine_fn=tf.multiply, name="%s_add_layer" % (name))

        return pheat


def AttentionPartsCRF(data,numin,lrnSize, iterSize, usepart,name=""):

    if usepart == 0:
        return AttentionIter(data,numin,lrnSize,iterSize=iterSize,name="%s_Attention"%(name))
    else:
        partnum = opt.partnum
        pre = []
        for i in range(partnum):
            att = AttentionIter(data=data,numin=numin,lrnSize=lrnSize, iterSize=iterSize,name="%s_Attention_%d"%(name,i))
            tmpconv = conv_2d(att,  1, filter_size=(1,1), name='%s_conv_%d' % (name,i))
            pre.append(tmpconv)

        return tl.layers.ConcatLayer(pre, -1)
        #return mx.symbol.concat(data=pre, dim=-1)
