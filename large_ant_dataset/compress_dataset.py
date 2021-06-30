import os

from multiprocessing import Pool

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/data'
DATASET = 'ant_dataset'


TAGGED = 'tagged'
VID_DIR = os.path.join(BASE_DIR, DATASET, TAGGED)
DEST_DIR = os.path.join(BASE_DIR, f'{DATASET}_compressed', TAGGED)


def convert_frames(vid_file, file_name):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    height, width = None, None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    while True:
        success, image = capture.read()
        if not success:
            break
            # raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        if read_count == 0:
            height, width, layers = image.shape
            out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))
        out.write(image)
        if read_count % 20 == 0:
            print(read_count)
        read_count += 1
    out.release()
    return frames


def process_train_vid_file(video_file):
    video_path = os.path.join(VID_DIR, video_file)
    output_path = os.path.join(DEST_DIR, video_file)
    convert_frames(video_path, output_path)


video_files = os.listdir(VID_DIR)
with Pool(6) as p:
    p.map(process_train_vid_file, video_files)