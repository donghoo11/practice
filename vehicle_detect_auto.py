import argparse
import json
import os
import shutil

import cv2
import glob
from model.models import VehicleDetectModel
from utils.utils import labelme_form, temp_shape


def main(cfg):
    data_path = sorted(glob.glob(f"{cfg.data_path}/*.{cfg.data_ext}"))

    # os.makedirs(f'frame_data/{cfg.data_path}/{cfg.data_path}_no_detect', exist_ok=True)

    vehicle_detecting_model = VehicleDetectModel(cfg.vehicle_detect_model_path, input_size=(416, 416), conf_thd=0.3,
                                                 iou_thd=0.3)

    for idx, file in enumerate(data_path):
        image = cv2.imread(file)
        basename = os.path.basename(file)

        org_H, org_W, _ = image.shape

        plate_car_bboxes = vehicle_detecting_model(image)

        label_dict = labelme_form()
        label_dict["imagePath"] = basename
        label_dict["imageHeight"] = org_H
        label_dict["imageWidth"] = org_W
        # if len(plate_car_bboxes) == 0:
        #     shutil.move(file, f'{cfg.data_path}/{cfg.data_path}_no_detect/{basename}')
        #     continue

        sort_plate_car_bboxes = sorted(plate_car_bboxes, key=lambda x: x[-1], reverse=True)

        for bbox in sort_plate_car_bboxes:
            box = list(map(float, bbox))
            label = "car" if box[-1] == 1 else "plate"
            data_shape = temp_shape(label)
            data_shape["points"] = [box[:2], box[2:4]]
            label_dict["shapes"].append(data_shape)

        json_path = file.replace(f".{cfg.data_ext}", ".json")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(label_dict, f, ensure_ascii=False, indent=4)

        print(f"\r진행 {(idx + 1) / len(data_path) * 100:.2f} %", end='')


if __name__ == '__main__':
    # frame에서 차량 및 번호판 박스 labelme 형식으로 auto labeling
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='vehicle_folder')
    parser.add_argument('--data_ext', type=str, default='jpg')
    parser.add_argument('--vehicle_detect_model_path', type=str, default='건설기계/vehicle_detecting_0916.onnx')
    config = parser.parse_args()
    main(config)
