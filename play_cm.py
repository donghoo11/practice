import cv2
import glob
import shutil
import copy
import os


image_path = sorted(glob.glob('건설기계/old_classify_dataset/valid/*h/*.jpg'))
i = 0
for idx, file in enumerate(image_path):
    img = cv2.imread(file)
    i += 1
    basename = os.path.basename(file)
    local_name = basename.split('_')[0]
    # cv2.imshow('yellow',img)
    # print(local_name)
    # os.makedirs(f'made_hc_plate/{local_name}', exist_ok=True)
    # shutil.copy(file,f'made_hc_plate/{local_name}/{basename}')
    # continue
    b = img.item(0, 0, 0)
    g = img.item(0, 0, 1)
    r = img.item(0, 0, 2)
    # cv2.imshow('t',img)
    print(b,g,r)
    if b+30 <= r:
        shutil.copy(file, f'prepare_cm_plate/yellow_h_image/{basename}')

    else:
        shutil.copy(file,f'prepare_cm_plate/green_h_image/{basename}')
    ord_H, ord_W, _ = img.shape

    # 노란색 글씨 변환
    # for y in range(ord_H):
    #     for x in range(ord_W):
    #         b = img.item(y, x, 0)
    #         g = img.item(y, x, 1)
    #         r = img.item(y, x, 2)
    #         # if g>=(r+60):
    #         #     # cv2.imshow('t',img)
    #         #     print(basename)
    #         #
    #         #     shutil.copy(file,f'green_h_image/{basename}')
    #         #     continue
    #
    # for y in range(ord_H):          #초록색
    #     for x in range(ord_W):
    #         b = img.item(y, x, 0)
    #         g = img.item(y, x, 1)
    #         r = img.item(y, x, 2)
            # if g>=(r+60):
            #     # cv2.imshow('t',img)
            #     print(basename)
            #
            #     shutil.copy(file,f'green_h_image/{basename}')
            #     continue
    # cv2.waitKey(0)
print(i)