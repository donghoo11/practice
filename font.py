from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import shutil
import cv2
import numpy as np
import textwrap


def make_info_image(img, text, img_size, fontpath, font_size):
    img_data = img.copy()
    # img_data = cv2.resize(img_data, (img_size[0], img_size[1]))
    # fontpath = "main_package/font/PureunJeonnam-Bold.ttf"
    font = ImageFont.truetype(fontpath, font_size)
    # text_img = np.zeros((int(img_size[1] / 2), img_size[0], 3), dtype=np.uint8)
    drawing = Image.fromarray(img_data.astype(np.uint8))
    draw = ImageDraw.Draw(drawing)
    if len(text) != 0:
        draw.text((0, 0), f"{text}", font=font, fill=(0, 0, 0))
    text_img = np.array(drawing, dtype=np.uint8)

    # info_img = cv2.vconcat([img_data, text_img])
    # info_img = cv2.resize(info_img, (img_size[0], img_size[1]))

    return text_img


def main(cfg):
    os.makedirs(f"font/{cfg.font}", exist_ok=True)
    names = list()
    with open(f"label.txt", 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            names.append(line)
    font_size = None
    font = None
    box = None
    for label in names:
        W, H = (200, 200)
        if 'h' in label:
            label = label.replace('h','')
            label = '   '.join(label)
            print(label)
            W, H = (600, 200)
        # if 'v' in label:
        #     label = label.replace('v','')
        #     label = '\n'.join(label)
        #     W, H = (200,400)

        image = Image.new('RGB', (W, H), (255, 255, 255))

        new_font = ImageFont.truetype(f"font/{cfg.font}.ttf", 200)
        draw = ImageDraw.Draw(image)


        new_box = draw.textbbox((0, 0), label, new_font)


        new_w = new_box[2] - new_box[0]
        new_h = new_box[3] - new_box[1]
        if new_w > W or new_h > H:
             print(label)
        font = new_font
        w = new_w
        h = new_h

        x = (W - w) / 2
        y = (H - h) / 2 -5

        draw.text((x, y), label, fill="black", font=font)
        image.save(f'{label}.jpg')

        # test_img = np.array(image, dtype=np.uint8)
        # cv2.imshow('t', test_img)
        # cv2.waitKey(0)
        shutil.move(f'{label}.jpg', f"font/{cfg.font}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', type=str, default='kor')
    config = parser.parse_args()
    main(config)

    # img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    #
    # img2 = make_info_image(img, "12351235가", (200, 200), 'PureunJeonnam-Bold.ttf', 10)
    # cv2.imshow('t', img2)
    # cv2.waitKey(0)

    # W = 640
    # H = 640
    # bg_color = 'rgb(214, 230,245)' #아이소프트존
    #
    # font = ImageFont.truetype("Balto-Medium.ttf", size=28)
    # font_color = 'rgb(0, 0, 0)'
    #
    # image = Image.new('RGB', (W, H), color= bg_color)
    # draw = ImageDraw.Draw(image)
    # w, h = font.getsize(message)
    # draw.text((50, 50), )
    # image.save('{}.png'.format(message))
    # image.show()
