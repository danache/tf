import tensorflow as tf


def accuracy_computation(pred,gtMap,batch_size,nstack,num_joint=14):
    joint_acc = []
    for i in range(num_joint):
        joint_acc.append(accurcy(pred[:,nstack - 1, :,:,i], gtMap[:,nstack - 1, :,:,i],batch_size))
    return joint_acc


def accurcy(pred, gtMap, num_img):
    err = tf.to_float(0)
    for i in range(num_img):
        err = tf.add(err, _compute_err(pred[i], gtMap[i]))
    return tf.subtract(tf.to_float(1), err / num_img)


def _compute_err(u, v):
    """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
	Args:
		u		: 2D - Tensor (Height x Width : 64x64 )
		v		: 2D - Tensor (Height x Width : 64x64 )
	Returns:
		(float) : Distance (in [0,1])
	"""
    u_x,u_y = argmax(u)
    v_x,v_y = argmax(v)
    return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))


def argmax(tensor):
    """ ArgMax
	Args:
		tensor	: 2D - Tensor (Height x Width : 64x64 )
	Returns:
		arg		: Tuple of max position
	"""
    resh = tf.reshape(tensor, [-1])
    argmax = tf.arg_max(resh, 0)
    return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])
