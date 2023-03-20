import random
import cv2
import glob
import numpy as np
import copy
import os
def emboss_edge(img):
    edge_img = cv2.Canny(img, 0, 255)               #윤곽선 구하기
    edge_img = np.stack([edge_img, edge_img, edge_img], axis=-1)
    la = 0.9
    img = cv2.addWeighted(img, la, edge_img, 1 - la, 0)
    return img
image_path = sorted(glob.glob('prepare_cm_plate/yellow_h_image/*.jpg'))
# image_path = sorted(glob.glob('test_dataset/jpg_json_valid_file/*.jpg'))
img_num = 0
for file in image_path:
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    num = 0
    basename = os.path.basename(file)
    org_H, org_W, _ = img.shape
    size = org_W * org_H
    start_b = img.item(0, 0, 0)
    start_g = img.item(0, 0, 1)
    start_r = img.item(0, 0, 2)
    for y in range(org_H):
        for x in range(org_W):
            h = hsv_img.item(y, x, 0)
            s = hsv_img.item(y, x, 1)
            v = hsv_img.item(y, x, 2)
            # print(h,s,v)
            start_h = hsv_img.item(0, 0, 0)
            start_s = hsv_img.item(0, 0, 1)
            start_v = hsv_img.item(0, 0, 2)
            # if start_s <120:
            #     if  v <130:
            #         hsv_img[y,x] = (0,0,255)
            # elif start_v <120:
            #     if v< 60:
            #         # print(h, s)
            #         hsv_img[y, x] = (0, 0, 255)
            # else:
            #     if v< 100:
            #         # print(h, s)
            #         hsv_img[y, x] = (0, 0, 255)
            # if start_s < 130:
            #     if v < 120:
            #         hsv_img[y,x] = (0,0,255)
            # else:
            if v<120:
                hsv_img[y,x] = (0,0,255)
                num += 1
                # print(b,g,r)

                # if start_b >=100:
                #     if (b <= 150) and (g <= 150) and (r <= 170):
                #         img_ex[y, x] = (200, 200, 200)
                # else:
                #     if (b<=50) and (g<=90) and (r<=170):
                #         img_ex[y,x] = (200,200,200)
        # hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    if size /5 < num and num < size /3:
        img_num += 1
        print('h', start_h)
        print('s', start_s)
        print('v', start_v)
        print('b', start_b)
        print('g', start_g)
        print('r', start_r)
        print(num/size)
        print(basename)
        cv2.imshow('org_img', img)
        #
        cv2.imshow('hsv_img', hsv_img)
    # cv2.waitKey(0)
    alpha = 0.2
    result = cv2.addWeighted(img, alpha, hsv_img, 1-alpha, 0)
    # cv2.imshow('flip',img_ex)
    result = cv2.bitwise_or(result, img)
    # cv2.imshow('bitwise',result)
    result = emboss_edge(result)
    # cv2.imshow('emboss', result)
    # print(type(result))
    img_array = np.asarray(hsv_img)
    img_array2 = np.asarray(result)

    kernel_size = random.randint(2, 5)
    kernel1d = cv2.getGaussianKernel(kernel_size, 3)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    hsv_img_res = cv2.filter2D(img_array, -1, kernel2d)
    result_res = cv2.filter2D(img_array2, -1, kernel2d)
    # print(type(all_img))
    # cv2.imshow('t4', hsv_img_res)
    # cv2.imshow('t5', result_res)

    cv2.waitKey(0)
    # print(img_num)

    # img_1 = img[0:org_H, 0:center_x]
    # img_2 = img[0:org_H, center_x:org_W]
    # img_ex_1 = result[0:org_H, 0:center_x]
    # img_ex_2 = result[0:org_H, center_x:org_W]
    # img_color = img[0:org_H, center_x - 1:center_x]
    # img_color_fin = img[0:org_H, center_x:center_x + 1]
    # img_ex_color = result[0:org_H, center_x - 1:center_x]
    # img_ex_color_fin = result[0:org_H, center_x:center_x + 1]
    # half_x = int(center_x / 2)
    # img_color = cv2.resize(img_color, (half_x, org_H))
    # img_color_fin =cv2.resize(img_color_fin, (half_x, org_H))
    # img_ex_color = cv2.resize(img_ex_color, (half_x, org_H))
    # img_ex_color_fin = cv2.resize(img_ex_color_fin, (half_x, org_H))
    #
    # # print(img_1.shape)
    # # print(img_2.shape)
    # # cv2.imshow('t', img_color)
    # # new = cv2.hconcat([img_ex_1, img_ex_2])
    # new_image = cv2.hconcat([img_1,img_color,img_color_fin,img_2])            # 그대로 합침 초록색
    # new_ex_image = cv2.hconcat([img_ex_1,img_ex_color,img_ex_color_fin,img_ex_2])       #img1,2 글씨색 변환 노란색
    # cv2.imshow('t4', new_ex_image)
    # cv2.waitKey(0)

    # print(img.shape)
    # print(mask.shape)
    # print(new.shape)
    # result = cv2.copyTo(img, mask, new)
    # cv2.imshow('t',result)
    # cv2.imshow('plate_hc',new_image)
    # cv2.imwrite(f'/home/kimdh/PycharmProject/practice/made_green_hc_plate/{basename}',new_image)       #초록색
    # cv2.imwrite(f'/home/kimdh/PycharmProject/practice/made_yellow_hc_plate/{basename}',new_ex_image)        #노란색
    # cv2.imshow('plate_yellow',new_ex_image)
    # cv2.imshow('green',new_image)
    # cv2.waitKey(0)


