import argparse
import glob
import json
import os
import random
import shutil
from multiprocessing import Pool

import cv2
import numpy as np

from cfg import prepare_data_dict


def prepare(file, dir='temp_data'):
    org_img = cv2.imread(file)
    org_H, org_W, _ = org_img.shape

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

    for i, shape in enumerate(shapes):
        label = shape["label"]
        points = shape["points"]
        points = np.array(points)

        save_path = f"{dir}/{label}/{label}_{plate_type}_{i}_{name}.jpg"

        minx, maxx = min(points[:, 0]), max(points[:, 0])
        miny, maxy = min(points[:, 1]), max(points[:, 1])

        # crop
        crop_img = org_img[int(miny):int(maxy), int(minx):int(maxx)]

        # classify
        cv2.imwrite(save_path, crop_img)


def main(cfg):
    # train valid split 전 임시 경로
    temp_data = 'temp_data'
    if not os.path.exists(temp_data):
        os.mkdir(temp_data)

    # 기존 dataset 지우기
    if cfg.reset and os.path.exists(cfg.dataset):
        shutil.rmtree(cfg.dataset)

    if not os.path.exists(cfg.dataset):
        os.mkdir(cfg.dataset)

    # train - valid dir 만들기
    train_dir = os.path.join(cfg.dataset, 'train')
    valid_dir = os.path.join(cfg.dataset, 'valid')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    label_names = list()
    with open(cfg.names_path, 'r', encoding='utf-8') as n:
        for line in n.readlines():
            label_names.append(line.strip())
    # class dir 만들기
    for label in label_names:
        temp_data_path = f"{temp_data}/{label}"
        train_path = f"{cfg.dataset}/train/{label}"
        valid_path = f"{cfg.dataset}/valid/{label}"

        for path in [temp_data_path, train_path, valid_path]:
            if not os.path.exists(path):
                os.mkdir(path)

    data_path = glob.glob(f'{cfg.data_path}/*.jpg')

    with Pool() as pool:
        pool.map(prepare, data_path)

    for label in label_names:
        temp_image_data_path = glob.glob(f'temp_data/{label}/*.jpg')

        count = int(len(temp_image_data_path) * cfg.valid_ratio)
        valid_data_list = random.sample(temp_image_data_path, count)  # valid set
        print(label, 'train :', len(temp_image_data_path) - count, 'valid :', count)

        # temp data to dataset
        for cls_img_file in temp_image_data_path:
            basename = os.path.basename(cls_img_file)
            if cls_img_file in valid_data_list:
                save_valid_data = f"dataset/valid/{label}/{basename}"
                shutil.move(cls_img_file, save_valid_data)
            else:
                save_train_data = f"dataset/train/{label}/{basename}"
                shutil.move(cls_img_file, save_train_data)

    shutil.rmtree(temp_data)


if __name__ == "__main__":
    default_cfg = prepare_data_dict

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=default_cfg["data_path"])
    parser.add_argument('--dataset', type=str, default=default_cfg['dataset'])
    parser.add_argument('--names_path', type=str, default=default_cfg["names_path"])
    parser.add_argument('--reset', type=bool, default=default_cfg["reset"])
    parser.add_argument('--valid_ratio', type=float, default=default_cfg["valid_ratio"])

    conf = parser.parse_args()

    main(conf)



# import argparse
# import glob
# import json
# import os
# import random
# import shutil
# from multiprocessing import Pool
# import copy
# import cv2
# import numpy as np
#
# from cfg import prepare_data_dict
#
# def yellow_letter_change(org_img):          # 추후 조금 더 수정 예정
#     org_H, org_W, _ = org_img.shape
#     hsv_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
#     img = copy.deepcopy(org_img)
#     for y in range(org_H):
#         for x in range(org_W):
#             h = hsv_img.item(y, x, 0)
#             s = hsv_img.item(y, x, 1)
#             v = hsv_img.item(y, x, 2)
#
#             if v < 100:
#                 hsv_img[y,x] = (0, 0, 200)
#     bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
#     alpha = 0.2
#     result = cv2.addWeighted(org_img, alpha, bgr_img, 1-alpha, 0)
#     img_array = np.asarray(result)
#
#     kernel_size = random.randint(2, 5)
#     kernel1d = cv2.getGaussianKernel(kernel_size, 3)
#     kernel2d = np.outer(kernel1d, kernel1d.transpose())
#     yellow_img = cv2.filter2D(img_array, -1, kernel2d)
#     return yellow_img
#
# def h_to_hc(org_img):
#     org_H, org_W, _ = org_img.shape
#     center_x = int(org_W / 2)
#     half_x = int(center_x / 2)
#     img_1 = org_img[0:org_H, 0:center_x]
#     img_2 = org_img[0:org_H, center_x:org_W]
#     img_center_1 = org_img[0:org_H, center_x - 1:center_x]
#     img_center_2 = org_img[0:org_H, center_x:center_x + 1]
#     img_center_1 = cv2.resize(img_center_1, (half_x, org_H))
#     img_center_2 = cv2.resize(img_center_2, (half_x, org_H))
#     hc_img = cv2.hconcat([img_1,img_center_1,img_center_2,img_2])
#     return hc_img
#
# def prepare(file, dir='temp_data'):
#     org_img = cv2.imread(file)
#     org_H, org_W, _ = org_img.shape
#
#     label_names = list()
#     with open('util/cls_local.names', 'r', encoding='utf-8') as n:
#         for line in n.readlines():
#             label_names.append(line.strip())
#
#     basename = os.path.basename(file)
#     name = os.path.splitext(basename)[0]
#
#     # labeled file load
#     json_path = file.replace('.jpg', '.json')
#
#     if not os.path.exists(json_path):
#         return False
#
#     with open(json_path, 'r', encoding='utf-8') as f:
#         json_file = json.load(f)
#
#     if 'check' in json_file['flags'].keys():
#         if json_file['flags']['check']:
#             return False
#
#     plate_type = ''
#     if 'green' in json_file['flags'].keys():
#         if json_file['flags']['green']:
#             plate_type = 'green'
#         elif json_file['flags']['yellow']:
#             plate_type = 'yellow'
#         else:
#             plate_type = 'normal'
#     shapes = json_file["shapes"]
#
#     loc_label = ''
#     loc_points = []
#     if len(shapes) > 8:
#         loc_shapes = copy.deepcopy(shapes[0])
#         loc_label = shapes[0]['label'] + shapes[1]['label'] + 'h'
#
#         loc_minx = shapes[0]['points'][0][0]
#         loc_maxx = shapes[1]['points'][1][0]
#         loc_miny = min(shapes[0]['points'][0][1], shapes[1]['points'][0][1])
#         loc_maxy = max(shapes[0]['points'][1][1], shapes[1]['points'][1][1])
#         loc_points.append([loc_minx,loc_miny])
#         loc_points.append([loc_maxx,loc_maxy])
#
#         loc_shapes['points'] = loc_points
#         loc_shapes['label'] = loc_label
#
#         shapes.append(loc_shapes)
#
#     for i, shape in enumerate(shapes):
#         label = shape["label"]
#         points = shape["points"]
#         points = np.array(points)
#
#         save_path = f"{dir}/{label}/{label}_{plate_type}_{i}_{name}.jpg"
#         save_path_hc = f"{dir}/{label}/{label}_{plate_type}_{i}_{name}_hc.jpg"
#
#         minx, maxx = min(points[:, 0]), max(points[:, 0])
#         miny, maxy = min(points[:, 1]), max(points[:, 1])
#
#         if label in label_names:
#             # crop
#             crop_img = org_img[int(miny):int(maxy), int(minx):int(maxx)]
#             hc_img = copy.deepcopy(crop_img)
#
#             if 'h' in label and maxx - minx < org_W / 3:
#                 if plate_type == 'yellow' and maxx-minx > 40:
#                     hc_img = yellow_letter_change(hc_img)
#
#                 hc_img = h_to_hc(hc_img)
#                 cv2.imwrite(save_path_hc, hc_img)
#         # classify
#         cv2.imwrite(save_path, crop_img)
#
#
# def main(cfg):
#     # train valid split 전 임시 경로
#     temp_data = 'temp_data'
#     if not os.path.exists(temp_data):
#         os.mkdir(temp_data)
#
#     # 기존 dataset 지우기
#     if cfg.reset and os.path.exists(cfg.dataset):
#         shutil.rmtree(cfg.dataset)
#
#     if not os.path.exists(cfg.dataset):
#         os.mkdir(cfg.dataset)
#
#     # train - valid dir 만들기
#     train_dir = os.path.join(cfg.dataset, 'train')
#     valid_dir = os.path.join(cfg.dataset, 'valid')
#
#     if not os.path.exists(train_dir):
#         os.mkdir(train_dir)
#     if not os.path.exists(valid_dir):
#         os.mkdir(valid_dir)
#
#     label_names = list()
#     with open(cfg.names_path, 'r', encoding='utf-8') as n:
#         for line in n.readlines():
#             label_names.append(line.strip())
#     # class dir 만들기
#     for label in label_names:
#         temp_data_path = f"{temp_data}/{label}"
#         train_path = f"{cfg.dataset}/train/{label}"
#         valid_path = f"{cfg.dataset}/valid/{label}"
#
#         for path in [temp_data_path, train_path, valid_path]:
#             if not os.path.exists(path):
#                 os.mkdir(path)
#
#     data_path = glob.glob(f'{cfg.data_path}/*.jpg')
#
#     with Pool() as pool:
#         pool.map(prepare, data_path)
#
#     for label in label_names:
#         temp_image_data_path = glob.glob(f'temp_data/{label}/*.jpg')
#
#         count = int(len(temp_image_data_path) * cfg.valid_ratio)
#         valid_data_list = random.sample(temp_image_data_path, count)  # valid set
#         print(label, 'train :', len(temp_image_data_path) - count, 'valid :', count)
#
#         # temp data to dataset
#         for cls_img_file in temp_image_data_path:
#             basename = os.path.basename(cls_img_file)
#             if cls_img_file in valid_data_list:
#                 save_valid_data = f"dataset/valid/{label}/{basename}"
#                 shutil.move(cls_img_file, save_valid_data)
#             else:
#                 save_train_data = f"dataset/train/{label}/{basename}"
#                 shutil.move(cls_img_file, save_train_data)
#
#     shutil.rmtree(temp_data)
#
#
# if __name__ == "__main__":
#     default_cfg = prepare_data_dict
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--data_path', type=str, default=default_cfg["data_path"])
#     parser.add_argument('--dataset', type=str, default=default_cfg['dataset'])
#     parser.add_argument('--names_path', type=str, default=default_cfg["names_path"])
#     parser.add_argument('--reset', type=bool, default=default_cfg["reset"])
#     parser.add_argument('--valid_ratio', type=float, default=default_cfg["valid_ratio"])
#
#     conf = parser.parse_args()
#
#     main(conf)
