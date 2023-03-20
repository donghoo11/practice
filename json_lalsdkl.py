
import argparse
import os
import shutil
import glob


def main(cfg):
    data_path = sorted(glob.glob(f"test_dataset/{cfg.data_dir}/*.{cfg.data_ext}"))
    for idx, file in enumerate(data_path):
        json_path = file.replace(f".{cfg.data_ext}",".json")
        basename = os.path.basename(file)


        if os.path.exists(json_path):
            shutil.move(file, f'test_dataset/cm_train_image/{basename}')
            basename_json = basename.replace(f".{cfg.data_ext}", ".json")
            shutil.move(json_path, f'test_dataset/cm_train_image/{basename_json}')
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='test_image')
    parser.add_argument('--data_ext', type=str, default='jpg')
    config = parser.parse_args()
    main(config)