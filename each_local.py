import json_lalsdkl
import os
import cv2
import glob
import argparse
import numpy as np
import copy
import shutil

def main(cfg):
    text_path = sorted(glob.glob(f'cm_dataset/labels/train/*.txt'))
    text_name = list()
    for idx, file in enumerate(text_path):
        basename = os.path.basename(file)

        if 'hc' in file:
            text_name.append(basename)

    text_file = sorted(glob.glob(f'txt/*.txt'))

    for idx, text in enumerate(text_file):
        txt_basename = os.path.basename(text)
        for text_list in text_name:
            if text_list in text:
                shutil.move(text, f'cm_dataset/labels/train/{txt_basename}')



    #
    # data_path = sorted(glob.glob(f"{cfg.data_dir}/*.{cfg.data_ext}"))
    # for idx, file in enumerate(data_path):
    #     json_path = file.replace(f'.{cfg.data_ext}','.json')
    #     basename = os.path.basename(file)
    #
    #     txt_path = basename.replace(f'.{cfg.data_ext}','')
    #
    #     img = cv2.imread(file)
    #     org_H, org_W, _ = img.shape
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         label_json = json.load(f)
    #     shapes = label_json["shapes"]
    #     # print(shapes)
    #     label = shapes[0]['label']
    #     fir_label = label[0:1]
    #     sec_label = label[1:2]
    #     sec_loc_label = copy.deepcopy(shapes[0])
    #     shapes.insert(1, sec_loc_label)
    #     shapes[0]['label'] = fir_label
    #     shapes[1]['label'] = sec_label
    #
    #     # print(shapes)
    #
    #     per = 0.3
    #     local_fir_points = shapes[0]["points"]
    #     local_sec_points = shapes[1]["points"]
    #     min_x, max_x = int(local_fir_points[0][0]), int(local_fir_points[1][0])
    #     w = max_x - min_x
    #     loc1_x = min_x + per * w
    #     loc2_x = max_x - per * w
    #     local_fir_points[1][0] = loc1_x
    #     local_sec_points[0][0] = loc2_x
    #
    #     with open(json_path, 'w', encoding='utf-8') as af:
    #         json.dump(label_json, af, ensure_ascii= False, indent=4)
    #
    #     label_points = list()
    #     for shape in shapes:
    #
    #         points = shape["points"]
    #         points = np.array(points)
    #         minx, maxx = min(points[:, 0]), max(points[:, 0])
    #         miny, maxy = min(points[:, 1]), max(points[:, 1])
    #
    #         center_x = (minx + maxx) / 2 / org_W
    #         center_y = (miny + maxy) / 2 / org_H
    #         W = abs(maxx-minx) / org_W
    #         H = abs(maxy-miny) / org_H
    #         label_points.append([0, center_x, center_y, W, H])
    #
    #     txt_file = f"txt/{txt_path}.txt"
    #
    #     with open(txt_file, 'w', encoding='utf-8') as t:
    #         for bbox in label_points:
    #             t.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")






    # h = 200
    # y1 = 200
    # y0 = 0
    # w = 600
    # x1 = int(0.3 * w)
    # x0 = 0
    # local_img = img[y0:y1, x0:x1]
    # cv2.imshow('img', local_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='cm_plate_0225')
    parser.add_argument('--data_ext', type=str, default='jpg')
    config = parser.parse_args()
    main(config)


