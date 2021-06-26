import os
import glob
from shutil import copyfile

base_dir = '/home/cabe0006/mb20_scratch/chamath/data/large_dataset/combined/videos'
output_dir = '/home/cabe0006/mb20_scratch/chamath/data/ant_dataset'
os.makedirs(output_dir, exist_ok=True)
tagged_dir = os.path.join(output_dir, 'tagged')
untagged_dir = os.path.join(output_dir, 'untagged')
os.makedirs(tagged_dir, exist_ok=True)
os.makedirs(untagged_dir, exist_ok=True)


def process_video_folder(dir_path, dir_name):
    save_dir = untagged_dir
    if dir_name[1] == 'T':
        save_dir = tagged_dir
    video_files = glob.glob(os.path.join(dir_path, '*.mp4'))
    for index, file in enumerate(video_files):
        copyfile(file, os.path.join(save_dir, f"{dir_name}_{index}.mp4"))


in_files = []
for dir in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir)
    sub_dirs = list(filter(lambda x: 'Over' not in x, os.listdir(dir_path)))

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(dir_path, sub_dir)
        in_files.append((sub_dir_path, sub_dir))
        # process_video_folder(sub_dir_path, sub_dir)
print(in_files)


