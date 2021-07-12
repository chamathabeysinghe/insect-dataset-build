import os
import pandas as pd
import cv2
import glob
import random
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,60)


def convert_and_save(file):
    img = cv2.imread(os.path.join(in_dir,file))
    name = file.split('.')[0]
    selected = df.loc[df['frame_id'] == name]
    for index, row in selected.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        w = int(row['w'])
        h = int(row['h'])
        cv2.rectangle(img, (x, y) , (x+w, y+h), (255, 0, 0) , 3)
    cv2.imwrite(os.path.join(out_dir, file), img)


tagged = 'tagged'
# base_path = "/Users/cabe0006/Projects/monash/Final_dataset/test_base"
base_path = "/home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images"
in_dir = os.path.join(base_path, f'{tagged}')
out_dir = os.path.join(base_path, f'out_{tagged}')
os.makedirs(out_dir, exist_ok=True)

annotation_path = os.path.join(base_path, 'records_corrected_False.csv')
df = pd.read_csv(annotation_path)

files = os.listdir(in_dir)
count = 30
selected_files = [files[random.randint(0, len(files) - 1)] for _ in range(count)]

for f in selected_files:
    convert_and_save(f)
