import cv2
import os
import glob


in_dir_base = "/home/cabe0006/mb20_scratch/chamath/data/evaluation_27/predictions/{}"
out_dir_base = "/home/cabe0006/mb20_scratch/chamath/data/evaluation_27/prediction_videos/{}"


def write_file(vid_frames, vid_file_name):
    height, width, layers = vid_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_file_name, fourcc, 6.0, (width, height))
    for i in range(len(vid_frames)):
        out.write(vid_frames[i])
#         out.write((vid_frames[i]).astype(np.uint8))
    out.release()


dirs = ['detr_v4_75']

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
for d in dirs:
    in_dir = "/home/cabe0006/mb20_scratch/chamath/data/evaluation_27/predictions/{}".format(d)
    out_dir = "/home/cabe0006/mb20_scratch/chamath/data/evaluation_27/prediction_videos/{}".format(d)
    os.makedirs(out_dir, exist_ok=True)
    for file_name in file_names:
        print(file_name)
        files = sorted(glob.glob(os.path.join(in_dir, f"{file_name}*")))
        images = [cv2.imread(f) for f in files]
        write_file(images, os.path.join(out_dir, f'{file_name}.mp4'))

