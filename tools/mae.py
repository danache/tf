
import json
import time
import argparse
import pprint

import numpy as np
delta= 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])


def load_annotations(anno_file):
    """Convert annotation JSON file."""

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                                       0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                                       0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:

        exit(0)

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def keypoint_eval(predictions, annotations):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0

    # Construct set to speed up id searching.
    prediction_id_set = set(predictions['image_ids'])
    # for every annotation in our test/validation set
    for image_id in prediction_id_set:
        # if the image in the predictions, then compute oks


        oks = compute_oks(anno=annotations['annos'][image_id], \
                          predict=predictions['annos'][image_id]['keypoint_annos'], \
                          delta=delta)
        # view pairs with max OKSs as match ones, add to oks_all
        oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
        # accumulate total num by max(gtN,pN)
        oks_num += np.max(oks.shape)


    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(oks_num))
    return np.mean(average_precision)

    #return return_dict


def getScore(predictions, anno):
    return keypoint_eval(predictions=predictions,
                                annotations=anno,
                                )


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))
    if predict_count == 0:
        return oks.T
    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = list(anno['keypoint_annos'].keys())[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = list(predict.keys())[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)

                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
    return oks

