import tensorflow as tf
import numpy as np
def MSE( output, target, nstack = 4, partnum=14,is_mean=False):
    print(output.get_shape())
    print(target.get_shape())

    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 5:  # [batch_size, n_feature]
            if is_mean:
                allloss = tf.Variable([0.],trainable=False,dtype=tf.float32)
                stack_loss = [0.]*nstack
                part_loss = [0.]*partnum

                for i in range(nstack):
                    for j in range(partnum):
                        tmp_ht = output[:,i,:,:,j]
                        tmp_gt = target[:,i,:,:,j]
                        tmp_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tmp_ht, tmp_gt )),
                                                       [1, 2])))
                        allloss += tmp_loss
                        #allloss,tmp_loss
                        stack_loss[i] +=tmp_loss
                        part_loss[j] += tmp_loss
                return (allloss, stack_loss, part_loss)
        #
        #
        #     else:
        #             res.append(tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output[i,:], target), [1, 2, 3, 4])))
        #     return mse
        # else:
        #     raise Exception("Unknow dimension")