# 1. 작업자가 완료한 image,json 파일을 모두 읽고
# 2. json이 있는지 없는지 판별
# 3. json이 있을 때 label(box or polygon)이 존재 판별

# 2,3에 대해 없는 경우 image,json 따로 다른 폴더로 옮기기
# 코드 X - image 검토

import argparse
import json_lalsdkl
import os
import shutil

import glob
from utils.utils import labelme_form


def main(cfg):
    # data_path = sorted(glob.glob(f"frame_data/{cfg.data_dir}/*.{cfg.data_ext}"))  # vehicle
    data_path = sorted(glob.glob(f"polygon_data/{cfg.data_dir}/*.{cfg.data_ext}")) # polygon
    os.makedirs(f'check/{cfg.data_dir}/no_object', exist_ok=True)

    for idx, file in enumerate(data_path):
        json_path = file.replace(f".{cfg.data_ext}", ".json")
        basename = os.path.basename(file)
        label_dict = labelme_form()
        label_dict["imagePath"] = basename

        if not os.path.exists(json_path):
            shutil.move(file, f'check/{cfg.data_dir}/no_object/{basename}')
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            label_json = json.load(f)

        if len(label_json["shapes"]) == 0:
            shutil.move(file, f'check/{cfg.data_dir}/no_object/{basename}')
            basename_json =basename.replace(f".{cfg.data_ext}",".json")
            shutil.move(json_path, f'check/{cfg.data_dir}/no_object/{basename_json}')
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='동대구_230103')
    parser.add_argument('--data_ext', type=str, default='png')
    config = parser.parse_args()
    main(config)
