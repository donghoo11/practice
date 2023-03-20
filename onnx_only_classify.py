import glob
import argparse
import onnx
import onnxruntime
import numpy as np
import cv2
from model.models import CharDetectModel, CharClassifyModel
from collections import Counter

def main(cfg):
    # cd_model = CharDetectModel('건설기계/detection_cm_each_230228.onnx',(416,416),0.2,0.2,0.01)
    cc_model = CharClassifyModel('건설기계/weight/classify_cm_class_230313.onnx', (28, 28))
    local_name = list()
    strings = ""
    with open('건설기계/cls_old.names', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            local_name.append(line.replace('\n', ''))
    acc = 0
    total = 0
    label_list = []
    pred_list = []
    for file in glob.glob(f"건설기계/cm_local_classify_dataset_h/valid/**/*.jpg"):
        basename = file.split('/')[3]
        total += 1
        predict_idx_list = []

        img = cv2.imread(file)
        pred_idx = cc_model(img)
        label_name = local_name[pred_idx]
        if label_name == basename:
            acc += 1
        else:
            label_list.append(f'{basename}'+'-'+f'{label_name}')
            # pred_list.append(label_name)
            print('label : ', basename)
            print('pred : ', label_name)
            # print(file)
            # cv2.imshow('t', img)
            #
            # cv2.waitKey(0)
        predict_idx_list.append(label_name)
    cnt = Counter(label_list)
    for key, value in cnt.items():

        strings += f"{key} : {value}\n"
    # with open(f'건설기계/wrong_data.txt','w') as f:
    #     f.write(strings)

    print(total - acc)
    print(total)
    print(acc / total * 100)
        # print(file)
        # char_bboxes =cd_model(img)
        #
        # char_img = img.copy()
        # predict_idx_list = []
        #
        # for bbox in char_bboxes:
        #     box = tuple(map(int,bbox[:4]))
        #     cv2.rectangle(img,box[:2],box[2:],(0,0,255),thickness=2)
        #
        #     char_classify_input=char_img[box[1]:box[3],box[0]:box[2]]
        #     pred_idx=cc_model(char_classify_input)
        #     label_name = local_name[pred_idx]
        #
        #     predict_idx_list.append(label_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='인')
    config = parser.parse_args()
    main(config)
