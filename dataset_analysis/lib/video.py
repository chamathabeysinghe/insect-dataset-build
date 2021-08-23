
def get_frames(vid_file, skip=1, max=-1):
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
        if read_count % skip == 0:
            frames.append(image)
        if read_count % 200 == 0:
            print(read_count)
        read_count += 1
        if read_count == max:
            break
    return frames