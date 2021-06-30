import os

from multiprocessing import Pool
import numpy as np

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/data'
# BASE_DIR = '/Users/cabe0006/Projects/monash/Datasets'
DATASET = 'ant_dataset'
# DATASET = 'ant_dataset_small'


TAGGED = 'tagged'
VID_DIR = os.path.join(BASE_DIR, DATASET, TAGGED)
DEST_DIR = os.path.join(BASE_DIR, f'{DATASET}_compressed', TAGGED)
os.makedirs(DEST_DIR, exist_ok=True)

def convert_frames(vid_file, file_name):
    print(file_name)
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    height, width = None, None
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    while True:
        success, image = capture.read()
        if not success:
            break
            # raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        if read_count == 0:
            height, width, layers = image.shape
            height = int(height/4)
            width = int(width/4)
            out = cv2.VideoWriter(file_name, fourcc, 12.0, (width, height))
        image = cv2.resize(image, (width, height))
        out.write(image.astype(np.uint8))

        if read_count % 20 == 0:
            print(read_count)
        read_count += 1
    out.release()
    return frames


def process_vid_file(video_file):
    video_path = os.path.join(VID_DIR, video_file)
    output_path = os.path.join(DEST_DIR, video_file)
    convert_frames(video_path, output_path)


video_files = os.listdir(VID_DIR)
print(video_files)
with Pool(6) as p:
    p.map(process_vid_file, video_files)