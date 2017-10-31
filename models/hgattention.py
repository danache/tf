from models.layers.AttentionPartsCRF import *
from models.layers.ResidualPool import *
from models.layers.Residual import *
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d
from tflearn.activations import relu,sigmoid
import numpy as np
import opt
def repResidual(data, num, nRep,name=""):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    out = []

    for i in range(nRep):
        if i == 0:
            tmpout = Residual(data,numin=num,numOut=num,name="%s_tmp_out%d"%(name,i))
        else:
            tmpout = ResidualPool(data=out[i - 1],numin=num,numOut=num,name="%s_tmp_out%d"%(name, i))
        out.append(tmpout)

    return out[-1]
def lin(data, numOut,name=None):
    with tf.variable_scope(name) as scope:

        conv1 = conv_2d(data, numOut, filter_size=(1,1), strides=(1,1),
                                name='conv1' )
        bn1 = tl.layers.BatchNormLayer(conv1,act=tf.nn.relu, name="bn1",)

        return bn1
def hourglass(data,n, f, imsize, nModual,name=""):
    #with mx.name.Prefix("%s_%s_" % (name, suffix)):
    pool =  tl.layers.MaxPool2d(data, (2,2),strides=(2,2),name='%s_pool1' %(name) )

    up = []
    low = []
    for i in range(nModual):

        if i == 0:
            if n > 1:
                tmpup = repResidual(data,num=f,nRep=n-1,name='%s_tmpup_'%(name) + str(i))
            else:
                tmpup = Residual(data,f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(pool,f,f,name='%s_tmplow_'%(name) + str(i))
        else:
            if n > 1:
                tmpup = repResidual(up[i-1],f, n - 1,name='%s_tmpup_'%(name) + str(i))
            else:
                tmpup = ResidualPool(up[i-1],f,f,name='%s_tmpup_'%(name) + str(i))
            tmplow = Residual(low[i - 1],f,f,name='%s_tmplow_'%(name) + str(i))

        up.append(tmpup)
        low.append(tmplow)

    if n > 1:
        low2 = hourglass(low[nModual - 1],n - 1,f,imsize / 2,nModual=nModual,name=name+"_" + str(n - 1)+"_low2")
    else:
        low2 = Residual(low[nModual - 1], f,f,name=name+"_"+str(n - 1)+"_low2")
    low3 = Residual(low2, f,f,name=name+"_"+str(n)+"low3")
    up2 = tl.layers.UpSampling2dLayer(low3, size=[2,2], is_scale = True,method=1,name="%s_Upsample"%(name))

    x = tl.layers.ElementwiseLayer(layer=[up[nModual - 1], up2],
                                   combine_fn=tf.add, name="%s_add_layer" % (name))

    return x

def createModel(data):
    #label = mx.symbol.Variable(name="hg_label")
    data = tl.layers.InputLayer(data, name='input')
    conv1 =  conv_2d(data, 64, filter_size=(7,7), strides=(1,1), padding='SAME',name="conv1")


    bn1 = tl.layers.BatchNormLayer(conv1,name="bn1",act=tf.nn.relu)

    r1 = Residual(bn1, 64,64,name="Residual1")

    pool1 = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

    r2 = Residual(pool1, 64,64,name="Residual2")

    r3 = Residual(r2, 64,128,name="Residual3")

    pool2 = tl.layers.MaxPool2d(r3, (2, 2), strides=(2, 2), name="pool2")
    r4 = Residual(pool2, 128,128,name="Residual4")

    r5 = Residual(r4, 128,128,name="Residual5")

    r6 = Residual(r5,128,opt.nFeats,name="Residual6")

    ####################################################
    #r6 = data
    ####################################################
    out = []
    inter = [r6]
    nPool = opt.nPool
    if nPool == 3:
        nModual = 16 / opt.nStack
    else:
        nModual = 8 / opt.nStack
    ###################################################test



    ##########################################################3
    for i in range(opt.nStack):

        hg = hourglass(data=inter[i],n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual),name="hourglass%d" %(i))#n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual))

        if i == opt.nStack - 1:
            ll1 = lin(hg,opt.nFeats * 2,name="hourglass%d_lin1"%(i))
            ll2 = lin(ll1,opt.nFeats * 2,name="hourglass%d_lin2"%(i))
            att = AttentionPartsCRF(ll2, opt.nFeats * 2,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
            tmpOut = AttentionPartsCRF(att, opt.nFeats * 2,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
        else:
            ll1 = lin(hg, opt.nFeats,name="hourglass%d_lin1"%(i))
            ll2 = lin(ll1, opt.nFeats,name="hourglass%d_lin2"%(i))

            if i >= 4 :
                att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))
                tmpOut = AttentionPartsCRF(att, opt.nFeats,opt.LRNKer, 3, 1,name="hourglass%d_attention2"%(i))
            else:

                att = AttentionPartsCRF(ll2, opt.nFeats,opt.LRNKer, 3, 0,name="hourglass%d_attention1"%(i))

                tmpOut = conv_2d(att, opt.partnum, filter_size=(1,1), strides=(1,1), name="hourglass%d_tmpout"%(i))

        out.append(tmpOut)

        if i < opt.nStack - 1:
            outmap = conv_2d(tmpOut, 256, filter_size=(1,1), strides=(1,1), padding='SAME',name="hourglass%d_outmap"%(i))
            ll3 = lin(outmap, opt.nFeats,name="hourglass%d_lin3"%(i))
            toointer =tl.layers.ElementwiseLayer(layer=[inter[i],outmap,ll3],
                                   combine_fn=tf.add, name="add_n%d"%(i))
            inter.append(toointer)
    # for i in range(8):
    #     out.append(np.random.rand(1, 14, 64, 64))

    return tl.layers.StackLayer(out, axis=1)
