import cv2
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,60)
import numpy as np
import random
import glob
import os
import cv2 as cv
from PIL import Image
import pandas as pd


base_dir = '/home/cabe0006/mb20_scratch/chamath/detr-v3'
syn_dir = os.path.join(base_dir, 'synthetic_dataset')
img_dir = os.path.join(base_dir, 'synthetic_dataset/train')
os.makedirs(img_dir, exist_ok=True)


def create_frame(background, ant_count, ant_imgs):
    image = Image.fromarray(background)
    csv_records = []
    for j in range(ant_count):
        #         print('j= {}'.format(j))
        rand_index = random.randint(0, len(ant_imgs) - 1)
        empty = np.zeros((background.shape[0], background.shape[1], 4)).astype(np.uint8)
        ant_img = ant_imgs[rand_index]
        a_w, a_h, _ = ant_img.shape

        #         rand_x = random.randint(10, background.shape[0] - a_w)
        #         rand_y = random.randint(10, background.shape[1] - a_h)
        rand_x = random.randint(10, background.shape[0] - a_w)
        rand_y = random.randint(background.shape[1] / 4, background.shape[1] * 3 / 4)
        empty[rand_x: rand_x + a_w, rand_y: rand_y + a_h, :] = ant_img
        overlay = Image.fromarray(empty)
        image.paste(overlay, mask=overlay)

        csv_records.append([rand_y, rand_x, a_h, a_w])

    return np.array(image).astype(np.uint8), csv_records


def generate_frames(background_name, ant_src, num_imgs, ant_count, results, file_index):
    background_path = '/home/cabe0006/mb20_scratch/chamath/detr-v3/background_images/images/{}'.format(
        background_name)
    background = cv2.imread(background_path)

    cropped_img_paths = glob.glob('/home/cabe0006/mb20_scratch/chamath/detr-v3/cropped_imgs/{}/*.png'.format(ant_src))
    ant_imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in cropped_img_paths]

    for i in range(num_imgs):
        print(i)
        file_name = f"A{ant_count}{background_name.split('.')[0]}_{file_index * num_imgs + i:06d}"
        img_path = os.path.join(img_dir, f'{file_name}.jpeg')
        frame, csv = create_frame(background, ant_count, ant_imgs)
        cv2.imwrite(img_path, frame)
        csv = list(map(lambda x: [file_name] + x, csv))
        results += csv
    return frame


ant_src = ['OU10B1L1Out_0', 'OU10B1L1In_0']
# ant_src = ['CU10L1B1In_0']

# background_names = ['b1_in.jpeg', 'b1_out.jpeg','b4_in.jpeg','b4_out.jpeg','b5_in.jpeg', 'b5_out.jpeg','b6_in.jpeg','b6_out.jpeg']
background_names = ['b1_in.jpeg', 'b1_out.jpeg','b5_in.jpeg', 'b5_out.jpeg']

results = []

file_index = 0
for background_name in background_names:
    for s in ant_src:
        for _ in range(1):
            count = 50
            print(background_name, s, count)
            generate_frames(background_name, s, 1, count, results, file_index)
            file_index += 1

df = pd.DataFrame(results, columns=['frame_id', 'x', 'y', 'w', 'h'])
df.to_csv(os.path.join(syn_dir, f'detections.csv'), index=False)







