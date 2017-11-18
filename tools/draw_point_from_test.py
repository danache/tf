import numpy as np
import cv2
import os
def draw_pic(heatmap, image_size, image_name,img_dir="",ori =None,thresh=0.2, pd=None):
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
            idx = np.unravel_index( single_data[last_stack,:,:,joint].argmax(), (256,256))

            visable = 1
            if single_data[last_stack,idx[0],idx[1],joint] < thresh:
                visable = 0

            res[joint][0] = idx[1]
            res[joint][1] = idx[0]
            res[joint][2] = visable

        ori_img = ori[batch,:]
        for i in range(heatmap.shape[-1]):
            if  res[i][2]  == 1:
                cv2.circle(ori_img, (int(res[i][0]), int(res[i][1])), 5, (0, 255, 155), -1)



        w, h, x1, y1, board_w, board_h, newsize_w, newsize_h = image_size[batch]

        if newsize_w < newsize_h:
            pad = np.int(newsize_h - newsize_w) * 0.5
            res[:,0] = np.maximum(res[:,0] - pad, 0)
        else:
            pad = np.int(newsize_w - newsize_h) * 0.5
            res[:, 1] = np.maximum(res[:, 1] - pad, 0)

        w_ratio = board_w * 1./ newsize_w

        h_ratio = board_h * 1./ newsize_h
        print("w_ratio" + str(w_ratio))
        print("h_ratio" + str(h_ratio))
        res[:, 0] = res[:, 0] * w_ratio + x1
        res[:, 1] = res[:, 1] * h_ratio + y1

        name = image_name[batch].decode("utf-8")
        img_path = os.path.join(img_dir, name + ".jpg")
        img = cv2.imread(img_path)
        for i in range(heatmap.shape[-1]):
            if res[i][2] == 1:
                cv2.circle(img, (int(res[i][0]), int(res[i][1])), 10, (0, 255, 155), -1)

        # cv2.imwrite("./img/%s.jpg" % name, img)
        cv2.imwrite("./img/Gauss_ori_%s.jpg" % name,ori_img )
        #



        row = pd[pd["image_id"] == name]

        #print(row["image_id"])
        img_path = os.path.join(img_dir, name + ".jpg")
        ori_anno = cv2.imread(img_path)
        keypoint = row["keypoint_annotations"].tolist()[0]


        anno = row["human_annotations"].tolist()[0]

        for key in anno.keys():

            ankle = keypoint[key].copy()
            print("gt, resize, *4")
            for i in range(14):
                if ankle[i * 3 + 2] == 1:
                    x = ankle[i * 3 ]
                    y =  ankle[i * 3 + 1]
                    cv2.circle(ori_anno, (int(x), int(y)), 5, (0, 255, 155), -1)
            ori_anno = cv2.resize(ori_anno,(256,256) )
            cv2.imwrite("./img/256%s.jpg" % name, ori_anno)



        # cv2.imwrite("./img/ori_anno_%s.jpg" % name, ori_anno)






