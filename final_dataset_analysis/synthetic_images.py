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
from multiprocessing import Pool

base_dir = '/Users/cabe0006/Projects/monash/Final_dataset'
syn_dir = os.path.join(base_dir, 'synthetic_dataset')
img_dir = os.path.join(base_dir, 'synthetic_dataset/images77')
os.makedirs(img_dir, exist_ok=True)


def increase_brightness(img_rgba, value=30):
    img = img_rgba[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    rimg = np.zeros((img.shape[0], img.shape[1], 4))
    rimg[:, :, 0:3] = img.astype(np.uint8)
    rimg[:, :, 3] = img_rgba[:, :, 3].astype(np.uint8)
    rimg = rimg.astype(np.uint8)
    return rimg


def resize(img):
    w, h, _ = img.shape
    #     mul = random.choice([1, .75, .8, 1.5, 1.25, 1.75, 2])
    mul = random.choice([1, .75, .8, 1.5, 1.25])
    #     mul = random.choice([1, 1.5, 1.25, 1.75, 2])

    new_w = int(mul * w)
    new_h = int(mul * h)
    return cv2.resize(img, (new_w, new_h))


def coloring(img_rgba):
    p = random.choice([0, 1, 2])
    return img_rgba
    if p == 0:
        return img_rgba
    elif p == 1:
        return brightness(img_rgba)
    elif p == 2:
        return darken(img_rgba)


def brightness(img_rgba):
    value = random.randint(30, 70)
    img = img_rgba[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    rimg = np.zeros((img.shape[0], img.shape[1], 4))
    rimg[:, :, 0:3] = img.astype(np.uint8)
    rimg[:, :, 3] = img_rgba[:, :, 3].astype(np.uint8)
    rimg = rimg.astype(np.uint8)
    return rimg


def darken(img_rgba):
    value = random.choice([x * 0.01 for x in range(50, 80)])
    img = img_rgba[:, :, :3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * value

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    rimg = np.zeros((img.shape[0], img.shape[1], 4))
    rimg[:, :, 0:3] = img.astype(np.uint8)
    rimg[:, :, 3] = img_rgba[:, :, 3].astype(np.uint8)
    rimg = rimg.astype(np.uint8)
    return rimg


def rotate(img_rgba):
    angle = [-1, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    p = random.choice(angle)
    if (p == -1):
        return img_rgba
    else:
        return cv2.rotate(img_rgba, p)


def create_frame(background, ant_count, ant_imgs):
    image = Image.fromarray(background)
    csv_records = []
    for j in range(ant_count):
        #         print('j= {}'.format(j))
        rand_index = random.randint(0, len(ant_imgs) - 1)
        empty = np.zeros((background.shape[0], background.shape[1], 4)).astype(np.uint8)
        ant_img = coloring(resize(rotate(ant_imgs[rand_index])))
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


combinations = [
    ('CU10L1B5In_4', 'b6_in.jpeg'),
    ('CU10L1B5In_4', 'b5_in.jpeg'),
    ('CU10L1B5In_4', 'b4_in.jpeg'),
    ('CU10L1B5In_4', 'b1_in.jpeg'),

    ('CU10L1B5Out_4', 'b6_out.jpeg'),
    ('CU10L1B5Out_4', 'b5_out.jpeg'),
    ('CU10L1B5Out_4', 'b4_out.jpeg'),
    ('CU10L1B5Out_4', 'b1_out.jpeg'),

    ('OU10B1L1In_0', 'b1_in.jpeg'),
    ('OU10B1L1In_0', 'b5_in.jpeg'),

    ('OU10B1L1Out_0', 'b1_out.jpeg'),
    ('OU10B1L1Out_0', 'b5_out.jpeg')
]
IMG_COUNT = 1
ANT_COUNT = 50


def process_combination(i):
    c = combinations[i]
    print(c)
    ant_src = c[0]
    background_name = c[1]
    background_path = '/Users/cabe0006/Projects/monash/Final_dataset/background_images/images/{}'.format(
        background_name)
    background = cv2.imread(background_path)
    cropped_img_paths = glob.glob('/Users/cabe0006/Projects/monash/Final_dataset/cropped_imgs/{}/*.png'.format(ant_src))
    ant_imgs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in cropped_img_paths]
    #     results = []
    for j in range(IMG_COUNT):
        file_name = f"A{ANT_COUNT}{background_name.split('.')[0]}_{i * IMG_COUNT + j:06d}"
        img_path = os.path.join(img_dir, f'{file_name}.jpeg')
        csv_path = os.path.join(img_dir, f'{file_name}.csv')

        frame, csv = create_frame(background, ANT_COUNT, ant_imgs)
        cv2.imwrite(img_path, frame)
        csv = list(map(lambda x: [file_name] + x, csv))
        df = pd.DataFrame(csv, columns=['frame_id', 'x', 'y', 'w', 'h'])
        df.to_csv(csv_path, index=False)


a = list(range(len(combinations)))
with Pool(5) as p:
    p.map(process_combination, a)

