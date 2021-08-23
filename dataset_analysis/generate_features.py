import os
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from lib.video import get_frames


video_base = '/Users/cabe0006/Projects/monash/Final_dataset/ant_dataset_compressed/untagged'


class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


res101_model = models.resnet101(pretrained=True)
res101_conv2 = ResNet50Bottom(res101_model)
res101_conv2.eval()
print('Model loading completed...')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
vid_files = os.listdir(video_base)
vid_files = sorted(list(filter(lambda x: '_0.mp4' in x, vid_files)))

with open('output.csv', 'w') as outfile:
    for f in vid_files:
        print(f)
        path = os.path.join(video_base, f)
        frames = get_frames(path, skip=50)
        N = len(frames)
        print(N)
        frames = np.asarray(frames).astype(np.float32)
        arr = torch.tensor(np.moveaxis(frames, -1, 1))
        arr = normalize(arr)
        arr_out = res101_conv2(arr)
        arr_out = arr_out.detach().numpy()
        arr_out = arr_out.reshape(N, -1)
        for arr in arr_out:
            line = ",".join([f] + list(map(str, arr)))
            outfile.write(line)
            outfile.write('\n')




