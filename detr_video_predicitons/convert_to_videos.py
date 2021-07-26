import cv2
import os
import glob


in_dir = "/home/cabe0006/mb20_scratch/chamath/detr-v3/evaluation_results/videos"
out_dir = "/home/cabe0006/mb20_scratch/chamath/detr-v3/evaluation_results/videos_mp4"
os.makedirs(out_dir, exist_ok=True)


def write_file(vid_frames, file_name):
    height, width, layers = vid_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))
    for i in range(len(vid_frames)):
        out.write(vid_frames[i])
#         out.write((vid_frames[i]).astype(np.uint8))
    out.release()


file_names = ["CU25L1B1In", "CU25L2B1Out", "OU50B1L1In", "OU50B1L1Out"]
for file_name in file_names:
    print(file_name)
    files = sorted(glob.glob(os.path.join(in_dir, f"{file_name}*")))
    images = [cv2.imread(f) for f in files]
    write_file(images, os.path.join(out_dir, f'{file_name}.mp4'))


