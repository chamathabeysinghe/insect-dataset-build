import pandas as pd
import json
import os

# BASE_DIR = '/Users/cabe0006/Projects/monash/Datasets/dataset-small'
# BASE_DIR = '/Users/cabe0006/Projects/monash/detr/evaluation/coco_evaluation/resource'
BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/detr-v3/synthetic_dataset'
# BASE_DIR = '/Users/cabe0006/Projects/monash/Final_dataset/dataset-small'

test_csv_path = os.path.join(BASE_DIR, 'detections.csv') #'/Users/cabe0006/Projects/monash/object-detection-v2/evaluation/coco_evaluation/resource/train.csv'
# test_csv_path = os.path.join(BASE_DIR, 'train.csv') #'/Users/cabe0006/Projects/monash/object-detection-v2/evaluation/coco_evaluation/resource/train.csv'
df = pd.read_csv(test_csv_path)
json_obj = {
    "categories": [
        {
        "id": 0,
        "name": "ant"
        }
    ],
    "images": [],
    "annotations": []
}

id_map = {}

for index, value in enumerate(df.frame_id.unique()):
    id_map[value] = index


for fileId in df.frame_id.unique():
    img_details = {
        "id": id_map[fileId],
        "license": 1,
        "file_name": "{}.jpg".format(fileId),
        "height": 2168,
        "width": 4096,
        "date_captured": "null"
    }
    json_obj["images"].append(img_details)

for index, row in df.iterrows():
    print()
    ant_details = {
        "id": index,
        "image_id": id_map[row['frame_id']],
        "category_id": 0,
        "bbox": [int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]),],
        "area": int(row["w"] * row["h"]),
          "iscrowd": 0
    }
    json_obj["annotations"].append(ant_details)

with open(os.path.join(BASE_DIR, 'ground-truth-train.json'), 'w') as outfile:
    json.dump(json_obj, outfile)

