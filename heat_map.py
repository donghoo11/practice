import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
from model.models import CharClassifyModel
import argparse

def main(cfg):
    cc_model = CharClassifyModel('건설기계/classify_cm_local_230310.onnx', (28, 28))
    local_name = list()
    with open('건설기계/cls_add_cm_local.names', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            local_name.append(line.replace('\n', ''))
    acc = 0
    total = 0
    label_list = list()
    pred_label = ""
    for file in glob.glob(f"건설기계/cm_local_classify_dataset/valid/**/*.jpg"):
        # print(file)
        basename = file.split('/')[3]

        total += 1
        predict_idx_list = []

        img = cv2.imread(file)
        # cv2.imshow('t',img)
        # cv2.waitKey(0)
        pred_idx = cc_model(img)
        if  pred_idx >= 94:
            pred_label = local_name[pred_idx]
            continue
        print(pred_label)
        # print(pred_idx)

        # print(label_name)

    # plt.Figure(figsize = (13, 13))
    # sns.heatmap(df,annot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    config = parser.parse_args()
    main(config)
