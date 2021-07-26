import cv2
import os

out_dir = "/home/cabe0006/mb20_scratch/chamath/data/evaluation_27/images_in"
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
        if read_count > 300:
            break
    return frames


def process(file_name):
#     file_name = "CU10L1B1Out_0"
    video_path = os.path.join(vid_dir, f'{file_name}.mp4')
    convert_frames(video_path, out_dir, file_name)


file_names = [
    "CU15L1B1In_1", "CU15L1B1Out_1",
    "CU15L1B4In_1", "CU15L1B4Out_1",
    "CU25L1B4In_1", "CU25L1B4Out_1",
    "CU10L1B5In_1", "CU10L1B5Out_1",
    "OU10B1L1In_1", "OU10B1L1Out_1",
    "OU10B3L3In_1", "OU10B3L3Out_1",
    "OU50B1L2In_1", "OU50B1L2Out_1",
    "OU50B2L1In_1", "OU50B2L1Out_1",
]
for file_name in file_names:
    print(file_name)
    process(file_name)



