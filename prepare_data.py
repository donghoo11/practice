import argparse
import glob
import json
import os
import random
import shutil
from multiprocessing import Pool
import cv2
import numpy as np
import copy

# from cfg import prepare_data_dict

def emboss_edge_yellow(img):
    org_H, org_W, _ = img.shape
    img_ex = copy.deepcopy(img)
    for y in range(org_H):
        for x in range(org_W):
            b = img.item(y, x, 0)
            g = img.item(y, x, 1)
            r = img.item(y, x, 2)
            start_b = img.item(0, 0, 2)
            if start_b >=100:
                if (b <= 100) and (g <= 150) and (r <= 170):
                    img_ex[y, x] = (200, 200, 200)
            else:
                if (b<=50) and (g<=90) and (r<=100):
                    img_ex[y, x] = (200,200,200)
    alpha = 0.2
    result = cv2.addWeighted(img, alpha, img_ex, 1-alpha, 0)
    img = cv2.bitwise_or(result, img)
    edge_img = cv2.Canny(img, 0, 255)               #윤곽선 구하기
    edge_img = np.stack([edge_img, edge_img, edge_img], axis=-1)
    la = 0.9
    img = cv2.addWeighted(img, la, edge_img, 1 - la, 0)

    img_array = np.asarray(img_ex)

    kernel_size = random.randint(2, 5)
    kernel1d = cv2.getGaussianKernel(kernel_size, 3)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    yellow_img = cv2.filter2D(img_array, -1, kernel2d)
    return yellow_img



def h_to_hc(img):
    org_H, org_W, _ = img.shape
    center_x = int(org_W / 2)
    center_y = int(org_H / 2)
    img_1 = img[0:org_H, 0:center_x]
    img_2 = img[0:org_H, center_x:org_W]
    img_center_1 = img[0:org_H, center_x - 1:center_x]
    img_center_2 = img[0:org_H, center_x:center_x + 1]
    half_x = int(center_x/2)
    img_center_1 = cv2.resize(img_center_1, (half_x, org_H))
    img_center_2 = cv2.resize(img_center_2, (half_x, org_H))
    hc_img = cv2.hconcat([img_1,img_center_1,img_center_2,img_2])
    return hc_img

def prepare(file, dir='temp_data'):
    org_img = cv2.imread(file)
    org_H, org_W, _ = org_img.shape
    local_name = list()
    with open('건설기계/cls_old.names', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            local_name.append(line.replace('\n',''))
    # cv2.imshow('tt',org_img)

    basename = os.path.basename(file)
    name = os.path.splitext(basename)[0]

    # labeled file load
    json_path = file.replace('.jpg', '.json')

    if not os.path.exists(json_path):
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)

    if 'check' in json_file['flags'].keys():
        if json_file['flags']['check']:
            return False

    plate_type = ''
    if 'green' in json_file['flags'].keys():
        if json_file['flags']['green']:
            plate_type = 'green'
        elif json_file['flags']['yellow']:
            plate_type = 'yellow'
        else:
            plate_type = 'normal'
    shapes = json_file["shapes"]

    loc_shapes = copy.deepcopy(shapes[0])
    loc_label = ''
    loc_point = []
    if len(shapes) > 8:
        loc_label = shapes[0]['label'] + shapes[1]['label'] + 'h'

        loc_minx = shapes[0]['points'][0][0]
        loc_maxx = shapes[1]['points'][1][0]
        loc_miny = min(shapes[0]['points'][0][1], shapes[1]['points'][0][1])
        loc_maxy = max(shapes[0]['points'][1][1], shapes[1]['points'][1][1])
        loc_point.append([loc_minx,loc_miny])
        loc_point.append([loc_maxx,loc_maxy])

        loc_shapes['points'] = loc_point
        loc_shapes['label'] = loc_label

        shapes.append(loc_shapes)
    # print(loc_shapes, org_W)
    # print(plate_type)
    for i, shape in enumerate(shapes):
        label = shape["label"]
        points = shape["points"]
        points = np.array(points)


        minx, maxx = min(points[:, 0]), max(points[:, 0])
        miny, maxy = min(points[:, 1]), max(points[:, 1])

        if label in local_name:
            save_path = f"{dir}/{label}/{label}_{plate_type}_{i}_{name}.jpg"
            save_path_hc = f"{dir}/{label}/{label}_{plate_type}_{i}_{name}_hc.jpg"

            # crop
            crop_img = org_img[int(miny):int(maxy), int(minx):int(maxx)]
            # cv2.rectangle(org_img,(minx,miny),(maxx,maxy),(0,0,255),thickness=2)
            # classify
            hc_img = copy.deepcopy(crop_img)

            if 'h' in label and maxx - minx < org_W / 3:
                if plate_type == 'yellow' and maxx-minx > 40:
                    hc_img = emboss_edge_yellow(hc_img)

                hc_img = h_to_hc(hc_img)
                cv2.imwrite(save_path_hc, hc_img)

            cv2.imwrite(save_path, crop_img)


def main(cfg):
    temp_data = 'temp_data'
    if not os.path.exists(temp_data):
        os.mkdir(temp_data)

    label_names = list()
    with open('건설기계/cls_old.names', 'r', encoding='utf-8') as n:
        for line in n.readlines():
            label_names.append(line.strip())
    print(label_names)
    for label in label_names:
        temp_data_path = f"{temp_data}/{label}"
        for path in [temp_data_path]:
            if not os.path.exists(path):
                os.mkdir(path)

    data_path = glob.glob(f'{cfg.data_path}/test_prepare/*.jpg')

    for file in data_path:
        prepare(file, dir=temp_data)
    for label in label_names:
        temp_image_data_path = glob.glob(f'temp_data/{label}/*.jpg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',type=str, default='test_dataset')
    conf = parser.parse_args()
    main(conf)