import pandas as pd
import cv2
import os
import random
import numpy as np


def convert_frames(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    # capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    while True:
        success, image = capture.read()
        if not success:
            break
            # raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        frames.append(image)

        if (read_count % 200 == 0):
            print(read_count)
        read_count += 1
    return frames


def write_file(vid_frames, file_name):
    height, width, layers = vid_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))

    for i in range(len(vid_frames)):
        out.write((vid_frames[i]).astype(np.uint8))
    out.release()


def visualize_df(df, file_name, vid_out_path):
    DIVIDER = 4.0
    # vid_path = os.path.join('/Users/cabe0006/Projects/monash/kalman_tracker/dataset/videos', f'{file_name}.mp4')
    vid_path = os.path.join('/Users/cabe0006/Projects/monash/Final_dataset/ant_dataset_compressed/untagged', f'{file_name}.mp4')
    frames = convert_frames(vid_path)
    colors = {}

    for id in df.track_id.unique():
        colors[id] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

    track_first_n_frames = len(df.image_id.unique())
    for i in range(track_first_n_frames):
        f = frames[i]
        boxes = df.loc[df['image_id'] == i]
        cv2.putText(f,
                    "Frame: " + str(i),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255),
                    1, cv2.LINE_AA)
        for index, row in boxes.iterrows():
            color = colors[row['track_id']]
            x = row['x'] / DIVIDER
            y = row['y'] / DIVIDER
            w = row['w'] / DIVIDER
            h = row['h'] / DIVIDER
            cv2.rectangle(f,
                          (int(x), int(y)),
                          (int(x + w), int(y + h)),
                          color, 2)
            cv2.putText(f,
                        str(int(row['track_id'])),
                        (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color,
                        1, cv2.LINE_AA)
    write_file(frames, vid_out_path)


base_dir = '/Users/cabe0006/Desktop/Experiments_2021_08_03/annotated_videos'
csv_dir = os.path.join(base_dir, 'csv')
video_dir = os.path.join(base_dir, 'videos')

files_names = os.listdir(csv_dir)
files_names = list(filter(lambda x: '.csv' in x, files_names))
files_names = [f.split('.')[0] for f in files_names]
files_names = ['CU15L1B4Out_0']
for file_name in files_names:

    print(file_name)
    # file_name = 'OU10B2L2In_0'

    # csv_path = os.path.join('/Users/cabe0006/Projects/monash/kalman_tracker/dataset/results', f'{file_name}.csv')
    csv_path = os.path.join(csv_dir, f'{file_name}.csv')
    vid_out_path = os.path.join(video_dir, f'{file_name}.mp4')
    DIVIDER = 4.0
    # csv_path = os.path.join('/Users/cabe0006/Projects/monash/kalman_tracker/dataset/temp', f'{file_name}.csv')
    df = pd.read_csv(csv_path)

    visualize_df(df, file_name, vid_out_path)
    # /Users/cabe0006/Downloads/OU10B1L1In_0.csv