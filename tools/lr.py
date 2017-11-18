import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
#
#decayed_learning_rate = learning_rate *
#                        decay_rate ^ (global_step / decay_steps)
#
def get_lr(last_learning_rate, global_step, decay_steps, decay_rate,human_decay,
                       name=None, ):

        p = int(global_step / decay_steps)

        return tf.multiply(tf.multiply(last_learning_rate,tf.pow(decay_rate, p)), human_decay, name=name
                           )
