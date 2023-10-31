import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_root', help='path to directory')
parser.add_argument('--out_root', help='path to directory')
parser.add_argument('--num_frames', type=int, help='the number of frames to extract', default=32)
args = parser.parse_args()

if not os.path.exists(args.out_root):
    os.mkdir(args.out_root)

to_iterate = list(os.walk(args.input_root))
for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
    for file in files:
        if file.endswith('.mp4'):
            image_path = os.path.join(root, file)
            vidcap = cv2.VideoCapture(image_path)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Fixed number of frames (same interval between frames)
            frame_idxs = np.linspace(0, frame_count - 1, args.num_frames, endpoint=True, dtype=int)

            out_path = image_path.replace(args.input_root, args.out_root)[:-4] + '/' # Cut .mp4 suffix
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            success, image = vidcap.read()
            count = 0
            while success:
                if count not in frame_idxs:
                    success, image = vidcap.read()
                    count += 1
                    continue
                cur_out_path = os.path.join(out_path, 'frame%d.jpg' % count)
                cv2.imwrite(cur_out_path, image)  # save frame as JPEG file
                success, image = vidcap.read()
                count += 1
            vidcap.release()