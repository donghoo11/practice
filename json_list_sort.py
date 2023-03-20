import json
import glob
import operator

data_path = sorted(glob.glob('test_vehicle_cm/*.json'))
for file in data_path:
    with open(file, 'r', encoding='utf-8') as f:
        label_json = json.load(f)

    for shape in label_json['shapes']:
        points = shape['points']
        pass
        # sorted_value = sorted(shape.items(), key=operator.itemgetter(), reverse=False)
        # print(sorted_value)



