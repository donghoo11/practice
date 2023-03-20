import argparse
import glob
import json
import os
import cv2

from model.models import PolygonDetectModel
from utils.utils import labelme_form, margin

def main(cfg):
    data_path = sorted(glob.glob(f'{cfg.data_dir}/*.{cfg.data_ext}'))

    os.makedirs(f'{cfg.data_dir}/polygon_0.4', exist_ok=True)
    polygon_detecting_model = PolygonDetectModel(cfg.model_path, input_size=(224,76))

    for idx, file in enumerate(data_path):
        image = cv2.imread(file)
        json_path = file.replace(f".{cfg.data_ext}", ".json")

        with open(json_path, 'r', encoding='utf-8') as f:
            label_json = json.load(f)

        basename = os.path.basename(file)
        org_name = os.path.splitext(basename)[0]

        plate_idx = 0

        for shape in label_json['shapes']:
            if shape["label"] != "plate":
                continue
            plate_idx +=1
            img_name = f"{org_name}_{plate_idx}.jpg"

            points = sum(shape["points"],[])
            crop_image = margin(image, points, ratio=cfg.crop_margin_ratio)

            crop_H, crop_W, _ = crop_image.shape

            edge_points =polygon_detecting_model(crop_image)

            label_dict = labelme_form()
            label_dict["imagePath"] = img_name
            label_dict["imageHeight"] = crop_H
            label_dict["imageWidth"] = crop_W
            label_dict['shapes'].append(
                {'label' : 'plate', 'points': edge_points.tolist(), 'group_id': None, 'shape_type': 'polygon','flags':{}}
            )

            save_img_path = f'{cfg.data_dir}/polygon_0.4/{img_name}'
            save_json_path = save_img_path.replace('.jpg','.json')

            cv2.imwrite(save_img_path,crop_image)

            with open(save_json_path, 'w', encoding='utf-8') as g:
                json.dump(label_dict, g, ensure_ascii=False, indent=4)


if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--data_dir', type=str, default='vehicle_folder')
      parser.add_argument('--data_ext', type=str, default='jpg')
      parser.add_argument('--crop_margin_ratio', type=tuple, default=(0.5,0.5))
      parser.add_argument('--model_path', type=str, default='건설기계/weight/poly_detect_RepVGG.onnx')
      config = parser.parse_args()
      main(config)