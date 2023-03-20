# 이름 변경시 json 내부 imagePath 이름 같게 만들어줌

import argparse
import json_lalsdkl
import os

import glob

def main(cfg):
    data_path = sorted(glob.glob(f"{cfg.data_dir}/*.{cfg.data_ext}"))  # polygon

    for idx, file in enumerate(data_path):
        basename = os.path.basename(file)  # .json 까지
        # img_path = file.replace('.json', '.jpg')
        name_jpg = basename.replace('.json', '.jpg')
        # print(name_jpg)

        with open(file, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
        # name_json = json_dict["imagePath"]
        # print(name_json)
        # json_dict["imagePath"].replace(name_json, name_jpg)
        json_dict["imagePath"] = name_jpg
        # json.replace(name_json, name_jpg)

        with open(file, 'w', encoding='utf-8') as sf:
            json.dump(json_dict, sf, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='plate_0223')
    parser.add_argument('--data_ext', type=str, default='json')
    config = parser.parse_args()
    main(config)
