import pandas as pd
import os

files = ['sample2.mp4', 'sample3.mp4']
# files = ['sample1.mp4', 'sample4.mp4','sample7.mp4','sample8.mp4','sample9.mp4','sample10.mp4','sample12.mp4','sample13.mp4','sample14.mp4',]
BASE_DIR = '/Users/cabe0006/Projects/monash/dataset-build/sample_data'
DATASET_BASE_DIR = '/Users/cabe0006/Projects/monash/dataset-build/sample_data'
INPUT_CSV_DIR = os.path.join(BASE_DIR, 'detection_data')
DETECTION_FILE = 'test.csv'
DATASET_DIR = os.path.join(DATASET_BASE_DIR, 'track_dataset')


track_first_n_frames = 10
start_index = 0
capture_len = 500

df = pd.read_csv(os.path.join(INPUT_CSV_DIR, DETECTION_FILE))

for f in files:
    VIDEO_INDEX = int(f[6: -4])
    csv_records = []
    image_ids = [VIDEO_INDEX*capture_len + i for i in range(capture_len)]
    dt_path = os.path.join(DATASET_DIR, f"sample{VIDEO_INDEX}", 'det')
    os.makedirs(dt_path, exist_ok=True)
    for i in range(track_first_n_frames):
        print(i)
        img_id = image_ids[i]
        boxes = df.loc[df['image_id'] == img_id].values.tolist()
        for box in boxes:
            record = [i + 1, -1, box[1], box[2], box[3], box[4], 1.0]
            csv_records.append(record)
    write_df = pd.DataFrame(csv_records)
    write_df.to_csv(os.path.join(dt_path, 'det.txt'), header=False, index=False)


