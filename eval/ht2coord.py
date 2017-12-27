import numpy as np
import os
import cv2
#img_dir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"
def getjointcoord(coord, img_name,predictions,thresh=0):


    for batch in range(coord.shape[0]):
        res = np.zeros([14,3])
        hm = np.squeeze(coord[batch,:])
        for joint in range(14):

            res[joint][0] = hm[joint][0]
            res[joint][1] = hm[joint][1]
            res[joint][2] = 0



        name = img_name[batch].decode("utf-8")
        #
        # img_path = os.path.join(img_dir, name + ".jpg")
        # img = cv2.imread(img_path)


        # for i in range(heatmap.shape[-1]):
        #     cv2.circle(img, (int(res[i][0]), int(res[i][1])), 10, (0, 255, 155), -1)
        # cv2.imwrite("./img/%s.jpg" % name, img)

        if name in predictions['image_ids']:
            num = len(predictions['annos'][name]['keypoint_annos'].keys()) + 1

            predictions['annos'][name]['keypoint_annos']['human%d'%num] = res
        else:
            predictions['image_ids'].append(name)
            predictions['annos'][name] = dict()
            predictions['annos'][name]['keypoint_annos'] = dict()
            predictions['annos'][name]['keypoint_annos']['human1'] = res
    return predictions



def getjointcoord2(heatmap, image_size, image_name,predictions,inputsize = 64,thresh=0):
    heatmap = np.array(heatmap)
    last_stack = heatmap.shape[1] - 1

    for batch in range(heatmap.shape[0]):
        res = np.ones(shape=(heatmap.shape[-1], 3)) * -1
        single_data = heatmap[batch,:]
        for joint in range(heatmap.shape[-1]):
            if inputsize == 256:
                idx = np.unravel_index(single_data[:, :, joint].argmax(), (inputsize, inputsize))
                visable = 1
                if single_data[ idx[0], idx[1], joint] < thresh:
                    visable = 0

                res[joint][0] = idx[1]
                res[joint][1] = idx[0]
                res[joint][2] = visable
            else:

                idx = np.unravel_index( single_data[last_stack,:,:,joint].argmax(), (inputsize,inputsize))
                visable = 1
                if single_data[last_stack,idx[0],idx[1],joint] < thresh:
                    visable = 0
                if inputsize == 64:
                    tmp_idx = np.asarray(idx) * 4
                res[joint][0] = tmp_idx[1]
                res[joint][1] = tmp_idx[0]
                res[joint][2] = visable

        w, h, x1, y1, board_w, board_h, newsize_w, newsize_h = image_size[batch]



        w_ratio = board_w * 1. / newsize_w

        h_ratio = board_h * 1. / newsize_h

        res[:, 0] = res[:, 0] * w_ratio + x1
        res[:, 1] = res[:, 1] * h_ratio + y1
        name = image_name[batch].decode("utf-8")
        #
        # img_path = os.path.join(img_dir, name + ".jpg")
        # img = cv2.imread(img_path)


        # for i in range(heatmap.shape[-1]):
        #     cv2.circle(img, (int(res[i][0]), int(res[i][1])), 10, (0, 255, 155), -1)
        # cv2.imwrite("./img/%s.jpg" % name, img)

        if name in predictions['image_ids']:
            num = len(predictions['annos'][name]['keypoint_annos'].keys()) + 1

            predictions['annos'][name]['keypoint_annos']['human%d'%num] = res
        else:
            predictions['image_ids'].append(name)
            predictions['annos'][name] = dict()
            predictions['annos'][name]['keypoint_annos'] = dict()
            predictions['annos'][name]['keypoint_annos']['human1'] = res
    return predictions



