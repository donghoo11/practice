
import glob

import onnx
import onnxruntime
import numpy as np
import cv2
from model.models import CharDetectModel, CharClassifyModel
import json
from collections import Counter

if __name__ == '__main__':
    cd_model = CharDetectModel('건설기계/detection_cm_each_230228.onnx',(416,416),0.2,0.3,0.01)
    cc_model = CharClassifyModel('건설기계/classify_cm_each_230302.onnx', (28, 28))
    local_name = list()
    acc = 0
    total_num = 0
    with open('건설기계/cls_add_cm_each.names', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            local_name.append(line.replace('\n',''))

    for file in glob.glob("test_dataset/cm_test_image/*.jpg"):
        img = cv2.imread(file)
        char_bboxes =cd_model(img)

        char_img = img.copy()
        num_list = []
        predict_idx_list = []
        json_path = file.replace(".jpg", ".json")

        with open(json_path, 'r', encoding='utf-8') as f:
            label_json = json.load(f)

        for shape in label_json["shapes"]:
            label = shape['label']
            num_list.append(label)

        for bbox in char_bboxes:
            box = tuple(map(int,bbox[:4]))
            cv2.rectangle(img,box[:2],box[2:],(0,0,255),thickness=2)

            char_classify_input=char_img[box[1]:box[3],box[0]:box[2]]
            pred_idx=cc_model(char_classify_input)
            label_name = local_name[pred_idx]

            predict_idx_list.append(label_name)
        cnt_num = Counter(sorted(num_list))
        cnt_pred = Counter(sorted(predict_idx_list))
        for key, value in cnt_num.items():
            if cnt_pred[key] == value:
                acc += value
            elif cnt_pred[key] < value:
                acc += cnt_pred[key]
        # print(cnt_num)
        # print(cnt_pred)
        # print(acc)
        for idx, file in enumerate(num_list):
            total_num += 1


            # if file in predict_idx_list:
            #     acc+=1
            # else:
            #     print(num_list)
            #     print(predict_idx_list)
            # if num_list[idx] == predict_idx_list[idx]:
            #     acc += 1

        cv2.imshow('t',img)
        print('label : ', num_list)
        print('pred : ', predict_idx_list)
        # print(acc)
        print(acc / total_num * 100)
        cv2.waitKey(0)

    # onnx_model = onnx.load("classify_add_cm_one_local230227.onnx")
    # onnx.checker.check_model(onnx_model)
    #
    # ort_session = onnxruntime.InferenceSession("classify_add_cm_one_local230227.onnx")
    #
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    #
    # img = cv2.imread('7_75.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, dsize=(28, 28))   # 416 x 416 x 3
    # img = img.astype(np.float32)
    # img = img/256.
    # img = np.transpose(img, (2,0,1))        # 3 X 416 X 416
    # img = img[np.newaxis, ...]      # 1 X 3 X 416 X 416
    # img = img.astype(np.float32)
    #
    # ort_inputs = {ort_session.get_inputs()[0].name: img}
    # ort_outs = ort_session.run(None, ort_inputs)
    # img_out = ort_outs[1]
    # out = np.argmax(img_out)
    # print(ort_outs)
    # print(out)





# img_in = img

# img_out = Image.fromarray(np.uint8((img_out[0]*255.0).clip(0, 255)[0]), mode='L')
#
# cv2.imshow('img', img_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()