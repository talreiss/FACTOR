from argparse import Namespace
from fairseq import checkpoint_utils, options, tasks, utils
import torch
import utils as avhubert_utils
from scipy.io import wavfile
from python_speech_features import logfbank
import numpy as np
import torch.nn.functional as F
import librosa
import os
from tqdm import tqdm
import argparse

def add_noise(clean_wav):
    def select_noise():
        noise_num = 1
        noise_wav = []
        rand_indexes = np.random.randint(0, len(noise_wav), size=noise_num)
        for x in rand_indexes:
            noise_wav.append(wavfile.read(noise_wav[x])[1].astype(np.float32))
        if noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav
    clean_wav = clean_wav.astype(np.float32)
    noise_snr = 0
    noise_wav = select_noise()
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1] + 1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav) / len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10 ** (snr / 20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed

def load_audio(path):
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
        return feats
    stack_order_audio: int=4
    noise_prob = 0
    # sample_rate, wav_data = wavfile.read(path)
    # wav_data, sample_rate = librosa.load(path)
    wav_data, sample_rate = librosa.load(path, sr=16_000)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    if np.random.rand() < noise_prob:
        wav_data = add_noise(wav_data)
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, stack_order_audio)  # [T/stack_order_audio, F*stack_order_audio]

    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats

def extract_visual_feature(model, video_path, audio_path):
    frames = avhubert_utils.load_video(video_path)
    frames = transform(frames)
    audio = load_audio(audio_path)[None,:,:].transpose(1, 2).cuda()
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    residual = frames.shape[2] - audio.shape[-1]
    # print(residual)
    if residual > 0:
        frames = frames[:, :, :-residual]
    elif residual < 0:
        audio = audio[:, :, :residual]
    # print(f"Audio shape is: {audio.shape}")
    # print(f"Video shape is: {frames.shape}")

    with torch.no_grad():
        # Specify output_layer if you want to extract feature of an intermediate layer
        feature_audio, _ = model.extract_finetune(source={'video': None, 'audio': audio}, padding_mask=None, output_layer=None)
        feature_audio = feature_audio.squeeze(dim=0)
        feature_vid, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
        feature_vid = feature_vid.squeeze(dim=0)
    return feature_audio.cpu().numpy(), feature_vid.cpu().numpy()


categories = ['rvra', 'rvfa', 'fvra', 'fvfa']
PATH_TO_DIRECTORY = '../../data/FakeAVCeleb_v1.2/'
ckpt_path = "../../misc/model.pt"  # LRS3
for category in categories:
    if category == 'rvra':
        input_root = os.path.join(PATH_TO_DIRECTORY, 'RealVideo-RealAudio')
    elif category == 'rvfa':
        input_root = os.path.join(PATH_TO_DIRECTORY, 'RealVideo-FakeAudio')
    elif category == 'fvra':
        input_root = os.path.join(PATH_TO_DIRECTORY, 'FakeVideo-RealAudio')
    elif category == 'fvfa':
        input_root = os.path.join(PATH_TO_DIRECTORY, 'FakeVideo-FakeAudio')

    to_iterate = list(os.walk(input_root))
    user_dir = os.getcwd()
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")
    model.cuda()
    model.eval()
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
    to_iterate = list(os.walk(input_root))
    all_audio, all_video, paths = [], [], []
    counter = 0
    for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
        for file in files:
            if file.endswith('.mp4') and not file.endswith('_roi.mp4'):
                counter += 1
                prefix = root.split('FakeAVCeleb_v1.2/')[1]
                mouth_roi_path = root + '/' + file[:-4] + '_roi.mp4'
                audio_path = root + '/' + file[:-4] + '.wav'
                try:
                    feature_audio, feature_vid = extract_visual_feature(model, mouth_roi_path, audio_path)
                except:
                    continue
                all_audio.append(feature_audio)
                all_video.append(feature_vid)
                paths.append(prefix + '/' + file)
    cur_split = input_root.split('/')[-1]
    all_audio = np.array(all_audio, dtype=object)
    all_video = np.array(all_video, dtype=object)
    paths = np.array(paths, dtype=object)
    cur_dir = os.path.join(user_dir, cur_split+'_features')
    if os.path.exists(cur_dir) is False:
        os.makedirs(cur_dir, exist_ok=True)
    print(f'{cur_split} number of videos: {counter}')
    print(f'{cur_split} shape of audio features: {all_audio.shape}')
    print(f'{cur_split} shape of video features: {all_video.shape}')
    np.save(f'{cur_dir}/audio.npy', all_audio)
    np.save(f'{cur_dir}/video.npy', all_video)
    np.save(f'{cur_dir}/paths.npy', paths)