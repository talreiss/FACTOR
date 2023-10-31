import sys
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('..')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory
import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
import numpy as np

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--data_conf_file", type = str,
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, default='AttentionNet',
                      help = "Resnet, Mobilefacenets.")
    conf.add_argument("--backbone_conf_file", type = str, default='./backbone_conf.yaml',
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type = int, default = 1024)
    conf.add_argument('--model_path', type = str, default = './Epoch_17.pt',
                      help = 'The path of model')
    conf.add_argument('--input_root', help = 'path to input directory after detection and alignment')
    args = conf.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = 127.5
    std = 128.0

    # define model.
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    model = model_loader.load_model(args.model_path).eval()

    path_names = []
    for root, d_names, f_names in tqdm(os.walk(args.input_root), total=len(list(os.walk(args.input_root)))):
        cur_vid_embeddings = []
        for frame in f_names:
            if not frame.endswith('.jpg'):
                continue
            p = frame
            path_names.append(p)
            file = os.path.join(root, frame)
            image = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (112, 112))
            image = (image.transpose((2, 0, 1)) - mean) / std
            image = torch.from_numpy(image.astype(np.float32))
            images = image[None].to(device)
            features = model(images)
            cur_vid_embeddings.append(features.detach().cpu().numpy())
        if len(cur_vid_embeddings) > 0:
            cur_vid_embeddings = np.concatenate(cur_vid_embeddings, 0)
            np.save(f'{root}/features.npy', cur_vid_embeddings)