import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np

from tools.draw_point_from_test import draw_pic

label_tmp = pd.read_json("/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json")
imgdir = "/media/bnrc2/_backup/ai/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/"
extension = 0
resize = 256
#label_tmp = pd.read_json(label_tmp)


def generateHeatMap(height, width, joints, num_joints, maxlenght):
    hm = []

    for i in range(int(num_joints)):
        tmp = (np.sqrt(maxlenght) * maxlenght * 10 / 4096.) + 2
        s = tmp
        x = joints[i * 3]
        y = joints[i * 3 + 1]

        if joints[i * 3 + 2] == 1.:

            ht = _makeGaussian(height, width, s, center=(x * 64,y * 64)),
        else:
            ht = planB(height, width)

        hm.append(ht)

    return hm


def _makeGaussian(height, width, sigma=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    # print(x,y)

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def planB(height, width):
    return np.zeros((height, width))



for index, row in label_tmp.iterrows():
    anno = row["human_annotations"]
    #         if(len(anno.keys())  == 1):
    #             continue
    img_path = os.path.join(imgdir, row["image_id"] + ".jpg")

    img = cv2.imread(img_path)

    w, h = img.shape[1], img.shape[0]
    keypoint = row["keypoint_annotations"]
    i = 0
    for key in anno.keys():
        i += 1
        if (anno[key][0] >= anno[key][2] or anno[key][1] >= anno[key][3]):
            print(img_path)
            continue

        x1, y1, x2, y2 = anno[key][0], anno[key][1], anno[key][2], anno[key][3]
        board_w = x2 - x1
        board_h = y2 - y1
        x1 = 0 if x1 - int(board_w * extension * 0.5) < 0 else x1 - int(board_w * extension * 0.5)
        x2 = w if x2 + int(board_w * extension * 0.5) > w else x2 + int(board_w * extension * 0.5)
        y1 = 0 if y1 - int(board_h * extension * 0.5) < 0 else y1 - int(board_h * extension * 0.5)
        y2 = h if y2 + int(board_h * extension * 0.5) > h else y2 + int(board_h * extension * 0.5)
        board_w = x2 - x1
        board_h = y2 - y1
        human = img[y1:y2, x1:x2]
        ankle = keypoint[key].copy()
        #             print(x1,y1,x2,y2)
        #             print(board_w,board_h)
        #             print(ankle)


        if board_h < board_w:
            newsize = (resize, board_h * resize // board_w)
        else:
            newsize = (board_w * resize // board_h, resize)
        for j in range(len(ankle)):
            if j % 3 == 0:
                ankle[j] = (ankle[j] - x1) / board_w
            elif j % 3 == 1:
                ankle[j] = (ankle[j] - y1) / board_h
            else:
                ankle[j] = ankle[j] * 1.

        # print(ankle)

        tmp = cv2.resize(human, newsize)
        new_img = np.zeros((resize, resize, 3))
        if (tmp.shape[0] < resize):  # 高度不够，需要补0。则要对item[6:]中的第二个值进行修改
            up = 0
            down =tmp.shape[0]
            new_img[up:down, :, :] = tmp
            for j in range(len(ankle)):
                if j % 3 == 1:
                    ankle[j] = (tmp.shape[0] * ankle[j] * 1. ) * 1./ resize
        elif (tmp.shape[1] < resize):
            left = 0
            right = tmp.shape[1]
            new_img[:, left:right, :] = tmp
            for j in range(len(ankle)):
                if j % 3 == 0:
                    ankle[j] = (tmp.shape[1] * ankle[j] * 1.) * 1./ resize

        heatmap = np.array(generateHeatMap( 64, 64, ankle, 14, 64))


        res = np.ones(shape=(14, 3)) * -1
        single_data = heatmap[0:]

        new_data = np.zeros([14, 1,256, 256])

        thresh = 0.3
        # print()
        for j in range(14):
            new_data[j, 0, :, :] = np.squeeze(
                cv2.resize(np.expand_dims(single_data[j,0, :, :], axis=-1), (256, 256)))

        single_data = new_data
        for joint in range(14):
            idx = np.unravel_index(single_data[joint, 0,:, :].argmax(), (256, 256))

            visable = 1


            res[joint][0] = idx[1]
            res[joint][1] = idx[0]

        print("______________")

        res = np.reshape(res, (42))
        #print()
        for i in range(len(ankle)):
            ankle[i] *= 256
        ankle = np.reshape(ankle,(42))
        print(np.abs(ankle -res) )
        print("______________")
    break
#             ori_img = ori[batch, :]
#             for i in range(heatmap.shape[-1]):
#                 if res[i][2] == 1:
#                     cv2.circle(ori_img, (int(res[i][0]), int(res[i][1])), 5, (0, 255, 155), -1)
#
#             w, h, x1, y1, board_w, board_h, newsize_w, newsize_h = image_size[batch]
#
#             if newsize_w < newsize_h:
#                 pad = np.int(newsize_h - newsize_w) * 0.5
#                 res[:, 0] = np.maximum(res[:, 0] - pad, 0)
#             else:
#                 pad = np.int(newsize_w - newsize_h) * 0.5
#                 res[:, 1] = np.maximum(res[:, 1] - pad, 0)
#
#             w_ratio = board_w * 1. / newsize_w
#
#             h_ratio = board_h * 1. / newsize_h
#             print("w_ratio" + str(w_ratio))
#             print("h_ratio" + str(h_ratio))
#             res[:, 0] = res[:, 0] * w_ratio + x1
#             res[:, 1] = res[:, 1] * h_ratio + y1
#
#             name = image_name[batch].decode("utf-8")
#             img_path = os.path.join(img_dir, name + ".jpg")
#             img = cv2.imread(img_path)
#             for i in range(heatmap.shape[-1]):
#                 if res[i][2] == 1:
#                     cv2.circle(img, (int(res[i][0]), int(res[i][1])), 10, (0, 255, 155), -1)
#
#             # cv2.imwrite("./img/%s.jpg" % name, img)
#             cv2.imwrite("./img/Gauss_ori_%s.jpg" % name, ori_img)
#             #
#
#
#
#             row = pd[pd["image_id"] == name]
#
#             # print(row["image_id"])
#             img_path = os.path.join(img_dir, name + ".jpg")
#             ori_anno = cv2.imread(img_path)
#             keypoint = row["keypoint_annotations"].tolist()[0]
#
#             anno = row["human_annotations"].tolist()[0]
#
#             for key in anno.keys():
#
#                 ankle = keypoint[key].copy()
#                 print("gt, resize, *4")
#                 for i in range(14):
#                     if ankle[i * 3 + 2] == 1:
#                         x = ankle[i * 3]
#                         y = ankle[i * 3 + 1]
#                         cv2.circle(ori_anno, (int(x), int(y)), 5, (0, 255, 155), -1)
#                 ori_anno = cv2.resize(ori_anno, (256, 256))
#                 cv2.imwrite("./img/256%s.jpg" % name, ori_anno)
#
#
# #np.array(list)