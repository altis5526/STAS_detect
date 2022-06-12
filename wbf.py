from tkinter import W
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import json 
import cv2
from ensemble_boxes import *
import argparse



def xywh2x1y1x2y2(bbox):
    x1 = bbox[0] - bbox[2]/2
    x2 = bbox[0] + bbox[2]/2
    y1 = bbox[1] - bbox[3]/2
    y2 = bbox[1] + bbox[3]/2
    return ([x1,y1,x2,y2])

def x1y1x2y22xywh(bbox):
    x = (bbox[0] + bbox[2])/2
    y = (bbox[1] + bbox[3])/2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return ([x,y,w,h])

def wbf(args):
    IMG_PATH = './STAS_dataset/Test_Images/'
    JSON_PATH = args.json_dir

    OUT_PATH = './result'


    MODEL_NAME = os.listdir(JSON_PATH)
    # MODEL_NAME = ['test1','test2']

    # ===============================
    # Default WBF config (you can change these)
    iou_thr = 0.67 #0.67
    skip_box_thr = 0.01
    # skip_box_thr = 0.0001
    sigma = 0.1
    # boxes_list, scores_list, labels_list, weights=weights,
    # ===============================

    dictionary = {}

    image_ids = os.listdir(IMG_PATH)
    for image_id in tqdm(image_ids, total=len(image_ids)):
        img = cv2.imread(IMG_PATH+image_id)
        h, w, _ = img.shape
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        for name in MODEL_NAME:
            box_list = []
            score_list = []
            label_list = []
            json_file = JSON_PATH + name
            if os.path.exists(json_file):
            # if os.path.getsize(txt_file) > 0:
                with open(json_file) as f:
                    try:
                        data = json.load(f)[image_id]
                    except:
                        data = []
                    for row in data:
                        row[0] = row[0]/w
                        row[1] = row[1]/h
                        row[2] = row[2]/w
                        row[3] = row[3]/h
                        box_list.append(row[0:4])
                        score_list.append(row[4])
                        label_list.append(0)
                boxes_list.append(box_list)
                scores_list.append(score_list)
                labels_list.append(label_list)
                weights.append(1.0)
            else:
                continue
                # print(txt_file)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        result = []
        for idx, box in enumerate(boxes):
            box[0] = int(box[0]*w)
            box[1] = int(box[1]*h)
            box[2] = int(box[2]*w)
            box[3] = int(box[3]*h)
            result.append( np.array(list(box) + [scores[idx]]))
        detect_flag = 0
        count_stas = 0
        for stas in result:
            xmin = int(stas[0])
            ymin = int(stas[1])
            xmax = int(stas[2])
            ymax = int(stas[3])
            confidence = float(stas[4])
            found_stas = np.array([[xmin, ymin, xmax, ymax, round(confidence,5)]])
            if confidence>=0.05:
                if count_stas==0:
                    final_stas = found_stas
                else:
                    final_stas = np.concatenate((final_stas, found_stas), axis=0)

                count_stas += 1
                detect_flag = 1
            
        if detect_flag:
            dictionary.update({f"{image_id}": None})
            dictionary[image_id] =  final_stas.tolist()

        for i in range(count_stas):
            dictionary[image_id][i][0] = int(dictionary[image_id][i][0])
            dictionary[image_id][i][1] = int(dictionary[image_id][i][1])
            dictionary[image_id][i][2] = int(dictionary[image_id][i][2])
            dictionary[image_id][i][3] = int(dictionary[image_id][i][3])

    json_object = json.dumps(dictionary)
    with open (args.output, 'w') as j:
        j.write(json_object)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, default="./label")
    parser.add_argument("--output", type=str, default="ensemble.json")
    args = parser.parse_args()

    wbf(args)


    