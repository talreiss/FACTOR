import numpy as np
from tqdm import tqdm
import faiss
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--category', default='rvfa')
args = parser.parse_args()

dir_RVRA = './RealVideo-RealAudio_features'
rvra_paths = np.load('../../data/rvra.npy')
category = args.category
if category == 'rvfa':
    dir_fake = './RealVideo-FakeAudio_features'
    fake_paths = np.load('../../data/rvfa.npy')
elif category == 'fvra':
    dir_fake = './FakeVideo-RealAudio_features'
    fake_paths = np.load('../../data/fvra_wl.npy')
elif category == 'fvfa-wl':
    dir_fake = './FakeVideo-FakeAudio_features'
    fake_paths = np.load('../../data/fvfa_wl.npy')
elif category == 'fvfa-fs':
    dir_fake = './FakeVideo-FakeAudio_features'
    fake_paths = np.load('../../data/fvfa_fs.npy')
elif category == 'fvfa-gan':
    dir_fake = './FakeVideo-FakeAudio_features'
    fake_paths = np.load('../../data/fvfa_gan.npy')
else:
    exit('Wrong category input')


rvra_audio = np.load(os.path.join(dir_RVRA, 'audio.npy'), allow_pickle=True)
rvra_vid = np.load(os.path.join(dir_RVRA, 'video.npy'), allow_pickle=True)
all_rvra_paths = np.load(os.path.join(dir_RVRA, 'paths.npy'), allow_pickle=True)

rvra_audio_video = []
lambda_percentile = 3

for i in tqdm(range(rvra_audio.shape[0]), total=rvra_audio.shape[0]):
    cur_path = all_rvra_paths[i]
    if cur_path not in rvra_paths:
        continue
    rvra_audio_unit = rvra_audio[i] / (np.linalg.norm(rvra_audio[i], ord=2, axis=-1, keepdims=True))
    rvra_vid_unit = rvra_vid[i] / (np.linalg.norm(rvra_vid[i], ord=2, axis=-1, keepdims=True))
    rvra_audio_video.append(np.percentile((rvra_audio_unit * rvra_vid_unit).sum(axis=1), lambda_percentile))

fake_audio = np.load(os.path.join(dir_fake, 'audio.npy'), allow_pickle=True)
fake_vid = np.load(os.path.join(dir_fake, 'video.npy'), allow_pickle=True)
all_fake_paths = np.load(os.path.join(dir_fake, 'paths.npy'), allow_pickle=True)

fake_audio_video = []

for i in tqdm(range(fake_audio.shape[0]), total=fake_audio.shape[0]):
    cur_path = all_fake_paths[i]
    if cur_path not in fake_paths:
        continue
    fake_audio_unit = fake_audio[i] / (np.linalg.norm(fake_audio[i], ord=2, axis=-1, keepdims=True))
    fake_vid_unit = fake_vid[i] / (np.linalg.norm(fake_vid[i], ord=2, axis=-1, keepdims=True))
    fake_audio_video.append(np.percentile((fake_audio_unit * fake_vid_unit).sum(axis=1), lambda_percentile))

test_set = -np.concatenate((rvra_audio_video, fake_audio_video))
labels = np.ones(len(test_set))
labels[:len(rvra_audio_video)] = 0

auc = roc_auc_score(labels, test_set)
ap = average_precision_score(labels, test_set)
print(f'AP: {ap*100}, AUC: {auc*100}')