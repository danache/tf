import numpy as np
import cv2
import os
def draw_pic(heatmap, image_size, image_name,predictions,img_dir="",ori =None,thresh=0.2, pd=None):
    heatmap = np.array(heatmap)
    last_stack = heatmap.shape[1] - 1
    ori = np.array(ori)
    for batch in range(heatmap.shape[0]):
        res = np.ones(shape=(heatmap.shape[-1], 3)) * -1

        single_data = np.array(heatmap[batch,:])

        new_data = np.zeros([1, 256, 256, single_data.shape[-1]])

        #print()
        for j in range(single_data.shape[-1]):


            new_data[0,:, :, j] =  np.squeeze(cv2.resize(np.expand_dims(single_data[last_stack,:,:,j], axis=-1),(256,256)))


        single_data = new_data
        for joint in range(heatmap.shape[-1]):
            idx = np.unravel_index( single_data[0,:,:,joint].argmax(), (256,256))

            visable = 1
            if single_data[0,idx[0],idx[1],joint] < thresh:
                visable = 0

            res[joint][0] = idx[1]
            res[joint][1] = idx[0]
            res[joint][2] = visable

        res1 = res.copy()
        w, h, x1, y1, board_w, board_h, newsize_w, newsize_h = image_size[batch]

        # if board_w* 1. == 255 * 1.:

        w_ratio = board_w * 1./ newsize_w

        h_ratio = board_h * 1./ newsize_h

        res[:, 0] = res[:, 0] * w_ratio + x1
        res[:, 1] = res[:, 1] * h_ratio + y1

        name = image_name[batch].decode("utf-8")



        if name in predictions['image_ids']:
            num = len(predictions['annos'][name]['keypoint_annos'].keys()) + 1

            predictions['annos'][name]['keypoint_annos']['human%d' % num] = res
        else:
            predictions['image_ids'].append(name)
            predictions['annos'][name] = dict()
            predictions['annos'][name]['keypoint_annos'] = dict()
            predictions['annos'][name]['keypoint_annos']['human1'] = res
        return predictions, res,res1