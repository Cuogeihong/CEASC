import numpy as np
import pickle
import json
import argparse

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_pathname', type=str)
    parser.add_argument('--json_pathname', type=str)
    return parser.parse_args()


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def main():
    args = parse_args()
    with open(args.pkl_pathname, 'rb') as pkl:
        data = pickle.load(pkl)

    results = []
    for i, pic_result in enumerate(data):
        img_id = i
        for idx, cate_result in enumerate(pic_result):
            result = np.array(cate_result)
            if result.shape[0] == 0:
                continue
            keep = py_cpu_nms(result,0.6)
            for bbox in result[keep]:
                x1, y1, x2, y2, score = bbox
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                pred_bbox = [x1,y1,x2,y2]
                image_result = {
                    'image_id': img_id,
                    'category_id': idx + 1,
                    'score': float(score),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                }
                results.append(image_result)
    json.dump(results, open(args.json_pathname, 'w'), indent=4)
    i = 1

if __name__ == '__main__':
    main()
