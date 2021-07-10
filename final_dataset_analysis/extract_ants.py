import cv2
import pandas as pd
import os
import json
import glob
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (60,20)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

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
#         if read_count == 10:
#             break
        read_count += 1
    return frames

def process_json_file(json_path, csv_records):
    with open(json_path) as f:
        data = json.load(f)
    for key in data:
        frame_id = key
        for box_key in data[key]:
            bbox = data[key][box_key]
            csv_records.append([frame_id] + bbox)


def process_file(file_name):
    annotation_path = os.path.join(base_dir, 'annotations', 'untagged', f'{file_name}.json')
    # video_path = os.path.join(base_dir, 'ant_dataset_compressed', 'untagged', f'{file_name}.mp4')
    video_path = os.path.join(base_dir, 'ant_dataset', 'untagged', f'{file_name}.mp4')
    frames = convert_frames(video_path)
    csv_records = []
    process_json_file(annotation_path, csv_records)

    extracts = {}
    for frame_index in range(0, len(frames), 10):
        filtered_records = list(filter(lambda x: f'{frame_index + 1:06d}' in x[0], csv_records))
        boxes = list(map(lambda x: {'x1': x[1], 'y1': x[2], 'x2': x[3] + x[1], 'y2': x[4] + x[2]}, filtered_records))
        #     print(boxes)
        extracts[frame_index] = []
        for i, b1 in enumerate(boxes):
            can_paint = True
            for j, b2 in enumerate(boxes):
                if (i == j):
                    continue
                iou_val = get_iou(b1, b2)
                if (iou_val > 0.0):
                    can_paint = False
                    break
            if can_paint:
                #             print("DLKJDLKFJHDLFKDJLFJDLFJDLKJFDLFJ")
                img = frames[frame_index]
                #             print(b1)
                extract = img[int(b1['y1'] / DIVIDER): int(b1['y2'] / DIVIDER),
                          int(b1['x1'] / DIVIDER): int(b1['x2'] / DIVIDER), :]
                if (0 in extract.shape):
                    continue
                cv2.imwrite(os.path.join(extracted_dir, f"{file_name}_{frame_index}_{i}.jpeg"), extract)
                extracts[frame_index].append(
                    (int(b1['x1'] / DIVIDER), int(b1['y1'] / DIVIDER),
                     int(b1['x2'] / DIVIDER), int(b1['y2'] / DIVIDER)))
                cv2.rectangle(img,
                              (int(b1['x1'] / DIVIDER), int(b1['y1'] / DIVIDER)),
                              (int(b1['x2'] / DIVIDER), int(b1['y2'] / DIVIDER)),
                              (255, 0, 0), 1)



DIVIDER = 1
# base_dir = "/Users/cabe0006/Projects/monash/Final_dataset"
base_dir = "/home/cabe0006/mb20_scratch/chamath/data"
extracted_dir = os.path.join(base_dir, 'cropped_imgs')
os.makedirs(extracted_dir, exist_ok=True)

files = ['CU10L1B1Out_0', 'CU10L1B1In_0', 'OU10B1L1In_0', 'OU10B1L1Out_0']

for file_name in files:
    process_file(file_name)
