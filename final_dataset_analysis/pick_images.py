import pandas as pd
import os
import cv2
import json
from multiprocessing import Pool
SKIP = 10

tagged = 'tagged'
video_dir = "/home/cabe0006/mb20_scratch/chamath/data/ant_dataset/tagged"
output_dir = "/home/cabe0006/mb20_scratch/chamath/object-detection-v3/dataset/train"


def convert_frames(vid_path):
    capture = cv2.VideoCapture(vid_path)
    read_count = 0
    print("Converting video file: {}".format(vid_path))
    frames = []
    while True:
        success, image = capture.read()
        if not success:
            break
            # raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        frames.append(image)
#         if read_count == 50:
#             break
#             print(read_count)
        read_count += 1
    return frames


def process_video(vid_file):
    frames = convert_frames(os.path.join(video_dir, f"{vid_file}.mp4"))
    for index in range(0, len(frames), SKIP):
        frame = frames[index]
        cv2.imwrite(os.path.join(output_dir, f"{vid_file}_{index+1:06d}.jpg"), frame)


video_files = os.listdir(video_dir)
video_files = list(filter(lambda x: '_0' in x, video_files))
video_files = [f.split('.')[0] for f in video_files]


with Pool(6) as p:
    p.map(process_video, video_files)

