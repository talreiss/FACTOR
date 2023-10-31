import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from IPython.display import HTML
from base64 import b64encode
import sys, subprocess
import argparse

def play_video(video_path, width=200):
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width={width} controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path+'/'+output_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in frames:
      landmark = detect_landmark(frame, detector, predictor)
      landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    try:
      rois = crop_patch(input_video_path+'/'+output_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                            window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    except:
      print(input_video_path+'/'+output_video_path)
      return
    roi_path = input_video_path + '/' + output_video_path[:-4] + '_roi.mp4'
    audio_fn = input_video_path + '/' + output_video_path[:-4] + '.wav'
    write_video_ffmpeg(rois, roi_path, "/usr/bin/ffmpeg")
    if '(East)' in input_video_path:
        input_video_path = input_video_path.replace(' (East)', '\ \(East\)')
        audio_fn = audio_fn.replace(' (East)', '\ \(East\)')
    elif '(South)' in input_video_path:
        input_video_path = input_video_path.replace(' (South)', '\ \(South\)')
        audio_fn = audio_fn.replace(' (South)', '\ \(South\)')
    elif '(American)' in input_video_path:
        input_video_path = input_video_path.replace(' (American)', '\ \(American\)')
        audio_fn = audio_fn.replace(' (American)', '\ \(American\)')
    elif '(European)' in input_video_path:
        input_video_path = input_video_path.replace(' (European)', '\ \(European\)')
        audio_fn = audio_fn.replace(' (European)', '\ \(European\)')
    cmd = "/usr/bin/ffmpeg" + " -i " + input_video_path+'/'+output_video_path + " -f wav -vn -y " + audio_fn + ' -loglevel quiet'
    subprocess.call(cmd, shell=True)
    # print(input_video_path + '/' + output_video_path)
    return

parser = argparse.ArgumentParser(description='')
parser.add_argument('--category', default='rvra')
args = parser.parse_args()
PATH_TO_DIRECTORY = '../../data/FakeAVCeleb_v1.2/'
category = args.category
face_predictor_path = "../../misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "../../misc/20words_mean_face.npy"
if category == 'rvra':
    input_root = os.path.join(PATH_TO_DIRECTORY, 'RealVideo-RealAudio')
elif category == 'rvfa':
    input_root = os.path.join(PATH_TO_DIRECTORY, 'RealVideo-FakeAudio')
elif category == 'fvra':
    input_root = os.path.join(PATH_TO_DIRECTORY, 'FakeVideo-RealAudio')
elif category == 'fvfa':
    input_root = os.path.join(PATH_TO_DIRECTORY, 'FakeVideo-FakeAudio')
else:
    exit('Wrong category input')

to_iterate = list(os.walk(input_root))
count = 0
for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
    flag = False
    for file in files:
        if file.endswith('.mp4') and not file.endswith('_roi.mp4'):
            count += 1
            preprocess_video(input_video_path=root, output_video_path=file, face_predictor_path=face_predictor_path,
                             mean_face_path=mean_face_path)