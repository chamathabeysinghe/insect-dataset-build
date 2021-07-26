import cv2
import os

out_dir = "/home/cabe0006/mb20_scratch/chamath/detr-v3/evaluation_results/images"
os.makedirs(out_dir, exist_ok=True)
vid_dir = "/home/cabe0006/mb20_scratch/chamath/data/ant_dataset/untagged"


def convert_frames(vid_path, out_dir, file_name):
    capture = cv2.VideoCapture(vid_path)
    read_count = 0
    print("Converting video file: {}".format(vid_path))
    frames = []
    while True:
        success, image = capture.read()
        if not success:
            break
        cv2.imwrite(os.path.join(out_dir, f"{file_name}_{read_count:06d}.jpeg"), image)
        read_count += 1
    return frames


def process(file_name):
#     file_name = "CU10L1B1Out_0"
    video_path = os.path.join(vid_dir, f'{file_name}.mp4')
    convert_frames(video_path, out_dir, file_name)


file_names = ["CU25L1B1In", "CU25L2B1Out", "OU50B1L1In", "OU50B1L1Out"]
for file_name in file_names:
    print(file_name)
    process(file_name)



