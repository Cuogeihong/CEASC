from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_pathname', type=str)
args = parser.parse_args()



train_info = '/path/to/dataset/annotations/instances_UAVval_v1.json'
json_file = args.json_pathname

json_gt = json.load(open(train_info))

json_result = json.load(open(json_file))

COCO_to_UAV = {

}

img_id_2_img_name = {}
img_id_2_img_results = {}
for i in range(len(json_gt['images'])):
    name = json_gt['images'][i]['file_name'][:-4] + '.txt'
    id = json_gt['images'][i]['id']
    img_id_2_img_name[id] = name

for item in json_result:
    bbox = item['bbox']
    score = item['score']
    image_id = item['image_id']
    category_id = item['category_id']
    if image_id not in img_id_2_img_results.keys():
        img_id_2_img_results[image_id] = []
    img_id_2_img_results[image_id].append(
        [bbox[0], bbox[1], bbox[2], bbox[3], score, category_id, -1, -1]
    )

root = './pred_txt/'

for img_id in img_id_2_img_results.keys():
    name = img_id_2_img_name[img_id]
    fp = open(root+name,'w')
    scores = []
    for item in img_id_2_img_results[img_id]:
        scores.append(item[4])
    for idx in np.argsort(scores)[::-1]:
        item = img_id_2_img_results[img_id][idx]
        outline = ''
        for num in item:
            outline += str(num) + ' '
        fp.write(outline + '\n')
    fp.close()

