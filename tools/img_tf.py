import tensorflow as tf


def unravel_argmax( ht, shape=([64, 64])):
    b = tf.reshape(ht, [-1])

    c = tf.cast(tf.argmax(b, 0), tf.int32)
    d = tf.cast(tf.div(c, tf.constant([64], dtype=tf.int32)), tf.int32)
    e = tf.subtract(c, tf.multiply(d, tf.constant([64], dtype=tf.int32)))
    return tf.stack([e,d])


def reverseFromHt(heatmap, nstack, batch_size, num_joint, scale, center, res=[64, 64]):
    scale = tf.cast(tf.squeeze(scale), tf.float32)
    center = tf.cast(tf.squeeze(center), tf.float32)

    end = []
    for n in range(batch_size):
        hm = heatmap[n, nstack - 1, :]
        res = []
        for joint in range(num_joint):
            idx = unravel_argmax(hm[:, :, joint])
            res.append(idx)
        res = tf.squeeze(tf.stack(res, axis=0))
        trans_coord = tf.cast(res,tf.float32)

        scale_t = tf.squeeze(scale[n, :])
        center_t = tf.squeeze(center[n,:])


        end.append(transformPreds(trans_coord,center_t ,scale_t , tf.constant([64., 64.]), reverse=1))

    return tf.stack(end,axis=0)

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = scale

    t00 = tf.divide(res[1], h[1])
    t11 = tf.divide(res[0], h[0])

    t02 = tf.multiply(res[1], (tf.add(tf.divide(-center[0], h[1]), .5)))
    t12 = tf.multiply(res[0], (tf.add(tf.divide(-center[1], h[0]), .5)))

    print
    t = tf.stack([[t00, 0, t02], [0, t11, t12], [0, 0, 1]])
    # [[0, t01, t02], [0, t11, t12], [0, 0, 1]],
    # t[0,1].assign(0)

    # if not rot == 0:
    #     rot = -rot # To match direction of rotation from cropping
    #     rot_mat = np.zeros((3,3))
    #     rot_rad = rot * np.pi / 180
    #     sn,cs = np.sin(rot_rad), np.cos(rot_rad)
    #     rot_mat[0,:2] = [cs, -sn]
    #     rot_mat[1,:2] = [sn, cs]
    #     rot_mat[2,2] = 1
    #     # Need to rotate around center
    #     t_mat = np.eye(3)
    #     t_mat[0,2] = -res[1]/2
    #     t_mat[1,2] = -res[0]/2
    #     t_inv = t_mat.copy()
    #     t_inv[:2,2] *= -1
    #     t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0, ):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)

    if invert:
        t = tf.matrix_inverse(t)

    p = tf.concat([pt, tf.ones([1], dtype=tf.float32)], axis=0)

    new_pt = tf.transpose(p)
    new_pt = tf.cast(new_pt, tf.float32)
    new_pt = tf.expand_dims(new_pt, -1)
    res = tf.matmul(t, new_pt)
    a = tf.stack([res[0], res[1]])
    return tf.cast(a, tf.float32)


def transformPreds(coords, center, scale, res,reverse=0):
    #     local origDims = coords:size()
    #     coords = coords:view(-1,2)
    lst = []

    for i in range(coords.shape[0]):
        lst.append(transform(coords[i], center, scale, res, reverse, 0, ))

    newCoords = tf.stack(lst, axis=0)

    return newCoords



def crop(img, h, w, center, scale, res, rot=0):
    # Upper left point
    ul = transform([0., 0.], center, scale, res, invert=1)
    # Bottom right point
    br = transform(res, center, scale, res, invert=1)

    # Padding so that when rotated proper amount of context is included
    #         pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    #         if not rot == 0:
    #             ul -= pad
    #             br += pad
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    x1 = tf.maximum(0., ul[0])
    y1 = tf.maximum(0., ul[1])
    sx = tf.subtract(tf.minimum(w, br[0]), tf.maximum(0., ul[0]))
    sy = tf.subtract(tf.minimum(h, br[1]), tf.maximum(0., ul[1]))
    size_x = tf.squeeze(
        tf.stack([tf.cast(sy, tf.int32), tf.cast(sx, tf.int32), tf.multiply(3, tf.ones(1, dtype=tf.int32))]))

    begin_x = tf.squeeze(tf.stack([y1, x1, tf.zeros(1)]))
    begin_x = tf.cast(begin_x, tf.int32)

    cpimg = tf.slice(img, begin_x, size_x)

    # if not rot == 0:
    #     # Remove padding
    #     new_img = scipy.misc.imrotate(new_img, rot)
    #     new_img = new_img[pad:-pad, pad:-pad]

    return tf.image.resize_images(cpimg, tf.cast(res, tf.int32))