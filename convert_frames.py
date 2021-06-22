import os


video_path = '/home/cabe0006/mb20_scratch/chamath/data/raw_videos'
dest_dir = '/home/cabe0006/mb20_scratch/chamath/data/frames'


files = ['sample15']
track_first_n_frames = 500


def convert_frames(vid_file, video_index, dest_dir):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    while read_count < track_first_n_frames:
        success, image = capture.read()
        if not success:
            raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        path = os.path.join(dest_dir, '{}.jpg'.format(read_count + video_index * track_first_n_frames))
        cv2.imwrite(path, image)
        if read_count % 20 == 0:
            print(read_count)
        read_count += 1


def process_train_vid_file(file, dest_dir):
    index = int(file.split('sample')[1])
    video_file = os.path.join(video_path, '{}.mp4'.format(file))
    convert_frames(video_file, index, dest_dir)


for file in files:
    dest_path = os.path.join(dest_dir, file)
    os.makedirs(dest_path, exist_ok=True)
    process_train_vid_file(file, dest_path)
