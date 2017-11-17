from hg_models.layers.AttentionPartsCRF import *
from hg_models.layers.ResidualPool import *
from hg_models.layers.Residual import *
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Conv2d as conv_2d

import numpy as np


class HGattention():
    def __init__(self, nFeat=256, nStack=8, nModules=1, outputDim=14,npool=4,lrnker=1
                 ):

        self.nStack = nStack
        self.nFeats = nFeat
        self.nModules = nModules
        self.partnum = outputDim
        self.npool = npool
        self.LRNKer = lrnker

    def repResidual(self,data, num, nRep,reuse=False,name=""):
        #with mx.name.Prefix("%s_%s_" % (name, suffix)):
        out = []

        for i in range(nRep):
            if i == 0:
                tmpout = Residual(data,numin=num,numOut=num,name="%s_tmp_out%d"%(name,i),reuse=reuse)
            else:
                tmpout = ResidualPool(data=out[i - 1],numin=num,numOut=num,name="%s_tmp_out%d"%(name, i),reuse=reuse)
            out.append(tmpout)

        return out[-1]
    def lin(self,data, numOut, reuse=False, name=None):
        with tf.variable_scope(name,reuse=reuse) as scope:

            conv1 = conv_2d(data, numOut, filter_size=(1,1), strides=(1,1),
                                    name='conv1' )
            bn1 = tl.layers.BatchNormLayer(conv1,act=tf.nn.relu, name="bn1",)

            return bn1
    def hourglass(self,data,n, f, imsize, nModual, reuse=False, name=""):
        #with mx.name.Prefix("%s_%s_" % (name, suffix)):
        with tf.variable_scope(name, reuse=reuse):
            pool =  tl.layers.MaxPool2d(data, (2,2),strides=(2,2),name='pool1' )

            up = []
            low = []
            for i in range(nModual):

                if i == 0:
                    if n > 1:
                        tmpup = self.repResidual(data,num=f,nRep=n-1,name='%s_tmpup_'%(name) + str(i),reuse=reuse)
                    else:
                        tmpup = Residual(data,f,f,name='%s_tmpup_'%(name) + str(i),reuse=reuse)
                    tmplow = Residual(pool,f,f,name='%s_tmplow_'%(name) + str(i),reuse=reuse)
                else:
                    if n > 1:
                        tmpup = self.repResidual(up[i-1],f, n - 1,name='%s_tmpup_'%(name) + str(i),reuse=reuse)
                    else:
                        tmpup = ResidualPool(up[i-1],f,f,name='%s_tmpup_'%(name) + str(i),reuse=reuse)
                    tmplow = Residual(low[i - 1],f,f,name='%s_tmplow_'%(name) + str(i),reuse=reuse)

                up.append(tmpup)
                low.append(tmplow)

            if n > 1:
                low2 = self.hourglass(low[nModual - 1],n - 1,f,imsize / 2,nModual=nModual,name=name+"_" + str(n - 1)+"_low2",reuse=reuse)
            else:
                low2 = Residual(low[nModual - 1], f,f,name=name+"_"+str(n - 1)+"_low2",reuse=reuse)
            low3 = Residual(low2, f,f,name=name+"_"+str(n)+"low3",reuse=reuse)
            up2 = tl.layers.UpSampling2dLayer(low3, size=[2,2], is_scale = True,method=1,name="Upsample")

            x = tl.layers.ElementwiseLayer(layer=[up[nModual - 1], up2],
                                           combine_fn=tf.add, name="%s_add_layer" % (name))

            return x

    def createModel(self,inputs,reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            data = tl.layers.InputLayer(inputs, name='input')
            #label = mx.symbol.Variable(name="hg_label")

            conv1 =  conv_2d(data, 64, filter_size=(7,7), strides=(1,1), padding='SAME',name="conv1")

            bn1 = tl.layers.BatchNormLayer(conv1,name="bn1",act=tf.nn.relu,)

            r1 = Residual(bn1, 64,64,name="Residual1",reuse=reuse)

            pool1 = tl.layers.MaxPool2d(r1, (2, 2), strides=(2, 2), name="pool1")

            r2 = Residual(pool1, 64,64,name="Residual2",reuse=reuse)

            r3 = Residual(r2, 64,128,name="Residual3",reuse=reuse)

            pool2 = tl.layers.MaxPool2d(r3, (2, 2), strides=(2, 2), name="pool2")
            r4 = Residual(pool2, 128,128,name="Residual4",reuse=reuse)

            r5 = Residual(r4, 128,128,name="Residual5",reuse=reuse)

            r6 = Residual(r5,128,self.nFeats,name="Residual6",reuse=reuse)

            ####################################################
            #r6 = data
            ####################################################
            out = []
            inter = [r6]
            nPool = self.npool
            if nPool == 3:
                nModual = 16 / self.nStack
            else:
                nModual = 8 / self.nStack
            ###################################################test



            ##########################################################3
            for i in range(self.nStack):

                hg = self.hourglass(data=inter[i],n=nPool,f=self.nFeats,imsize=self.partnum,nModual=int(nModual),name="hourglass%d" %(i),reuse=reuse)#n=nPool,f=opt.nFeats,imsize=opt.outputRes,nModual=int(nModual))

                if i == self.nStack - 1:
                    ll1 = self.lin(hg,self.nFeats * 2,name="hourglass%d_lin1"%(i),reuse=reuse)
                    ll2 = self.lin(ll1,self.nFeats * 2,name="hourglass%d_lin2"%(i),reuse=reuse)
                    att = AttentionPartsCRF(ll2, self.nFeats * 2,self.LRNKer, 3, 0,name="hourglass%d_attention1"%(i),partnum=self.partnum,reuse=reuse)
                    tmpOut = AttentionPartsCRF(att, self.nFeats * 2,self.LRNKer, 3, 1,name="hourglass%d_attention2"%(i),partnum=self.partnum,reuse=reuse)
                else:
                    ll1 = self.lin(hg, self.nFeats,name="hourglass%d_lin1"%(i),reuse=reuse)
                    ll2 = self.lin(ll1, self.nFeats,name="hourglass%d_lin2"%(i),reuse=reuse)

                    if i >= 4 :
                        att = AttentionPartsCRF(ll2, self.nFeats,self.LRNKer, 3, 0,name="hourglass%d_attention1"%(i),partnum=self.partnum,reuse=reuse)
                        tmpOut = AttentionPartsCRF(att, self.nFeats,self.LRNKer, 3, 1,name="hourglass%d_attention2"%(i),partnum=self.partnum,reuse=reuse)
                    else:

                        att = AttentionPartsCRF(ll2, self.nFeats,self.LRNKer, 3, 0,name="hourglass%d_attention1"%(i),partnum=self.partnum,reuse=reuse)

                        tmpOut = conv_2d(att, self.partnum, filter_size=(1,1), strides=(1,1), name="hourglass%d_tmpout"%(i))

                out.append(tmpOut)

                if i < self.nStack - 1:
                    outmap = conv_2d(tmpOut, 256, filter_size=(1,1), strides=(1,1), padding='SAME',name="hourglass%d_outmap"%(i))
                    ll3 = self.lin(outmap, self.nFeats,name="hourglass%d_lin3"%(i),reuse=reuse)
                    toointer =tl.layers.ElementwiseLayer(layer=[inter[i],outmap,ll3],
                                           combine_fn=tf.add, name="add_n%d"%(i))
                    inter.append(toointer)
            # for i in range(8):
            #     out.append(np.random.rand(1, 14, 64, 64))

            return tl.layers.StackLayer(out, axis=1,name='final_output')
        #end = tl.layers.StackLayer(out, axis=1, name='final_output')
