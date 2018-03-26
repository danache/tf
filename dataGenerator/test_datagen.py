import numpy as np
import os
import scipy.misc as scm
import json
import scipy

import tqdm
class DataGenerator():
    def __init__(self, json=None, img_path="", resize=256, normalize=True,
                 pixel_mean=np.array([[[102.9801, 115.9465, 122.7717]]])):
        self.img_dir = img_path
        self.json_path = json
        self.resize = [resize, resize]
        self.res = [64, 64]
        self.pixel_means = pixel_mean
        self.normalize = normalize
        self.load_data()

    def load_data(self):
        with open(self.json_path, "r") as f:
            json_file = json.load(f)
        human_lst = []
        for files in json_file:
            human_anno = files["human_annotations"]

            for human in human_anno.keys():
                human_lst.append(dict(box=human_anno[human],
                                      name=files['image_id']))

        self.test_data = human_lst

    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        img = scipy.misc.imread(os.path.join(self.img_dir,name))
        return img

    def get_transform(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / (h[1])
        t[1, 1] = float(res[0]) / (h[0])
        t[0, 2] = res[1] * (-float(center[0]) / h[1] + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h[0] + .5)
        t[2, 2] = 1
        return t

    def getN(self):
        return len(self.dataset)

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        # Transform pixel location to different reference
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)

    def transformPreds(self, coords, center, scale, res, reverse=0):
        #     local origDims = coords:size()
        #     coords = coords:view(-1,2)
        lst = []

        for i in range(coords.shape[0]):
            lst.append(self.transform(coords[i], center, scale, res, reverse, 0, ))

        newCoords = np.stack(lst, axis=0)

        return newCoords

    def crop(self, img, center, scale, res):

        ul = np.array(self.transform([0, 0], center, scale, res, invert=1))

        br = np.array(self.transform(res, center, scale, res, invert=1))

        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        new_img = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

        return scipy.misc.imresize(new_img, res)

    def getFeature(self, box):
        x1, y1, x2, y2 = box
        center = np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        scale = y2 - y1, x2 - x1
        return center, scale

    def recoverFromHm(self, hm, center, scale):
        res = []

        tmp_lst = []
        for i in range(hm.shape[-1]):
            index = np.unravel_index(hm[0, :, :, i].argmax(), self.res)
            tmp_lst.append(index[::-1])
        return self.transformPreds(np.stack(tmp_lst), center, scale, self.res, reverse=1)


    def get_batch_generator(self):
        idx = 0
        while idx < len(self.test_data):
            data_slice = self.test_data[idx]
            img = self.open_img(data_slice['name'])

            box = data_slice['box']
            center, scale = self.getFeature(box)

            crop_img = self.crop(img, center, scale, self.resize)
            crop_img = (crop_img.astype(np.float64) - self.pixel_means)
            if self.normalize:
                train_img = crop_img.astype(np.float32) / 255
            else:
                train_img = crop_img.astype(np.float32)
            train_img = np.expand_dims(train_img,0)
            yield train_img, img, center, scale,data_slice['name']
            idx += 1
            if idx % 50 == 0:
                print("%.2f "%(idx / len(self.test_data) * 100) +"% have done!!")