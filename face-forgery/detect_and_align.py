from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
import math
import sys, os
import cv2
import yaml
import numpy as np
from tqdm import tqdm
import argparse

def distance2center(x1, y1, x2, y2, image):
    im_cx = int(image.shape[1] / 2)
    im_cy = int(image.shape[0] / 2)
    cx = ((x2 + x1) / 2).astype(int)
    cy = ((y2 + y1) / 2).astype(int)
    return math.sqrt(math.pow(im_cx - cx, 2) + math.pow(im_cy - cy, 2))


def Filter2centerBox(boundingBoxes, frame):
    min_distance = 999999999
    min_idx = -1
    for i, det in enumerate(boundingBoxes):
        distance = distance2center(det[0], det[1], det[2], det[3], frame)
        if distance < min_distance:
            min_idx = i
            min_distance = distance
    return np.array([boundingBoxes[min_idx]]), len(boundingBoxes) > 1  # Added multi-face flag


def AlignedOneImageUsingFaceXAlignment(input_root, out_root, image_path, log_file):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        input_height, input_width, _ = image.shape
    except:
        return
    dets = faceDetModelHandler.inference_on_image(image)
    out_path = image_path.replace(input_root, out_root)
    
    if len(dets) > 0:
        dets, is_multi_face = Filter2centerBox(dets, image)
        
        if is_multi_face and log_file:
            log_file.write(f"Multi-face detected: {image_path}\n")
            
        for i, det in enumerate(dets):
            landmarks = faceAlignModelHandler.inference_on_image(image, det)
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks.reshape(-1))
            if os.path.exists(os.path.dirname(out_path)) is False:
                os.makedirs(os.path.dirname(out_path))
            cv2.imwrite(out_path, cropped_image)
    else:
        if os.path.exists(os.path.dirname(out_path)) is False:
            os.makedirs(os.path.dirname(out_path))
        cv2.imwrite(out_path, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_root', help='path to input directory')
    parser.add_argument('--out_root', help='path to output directory')
    args = parser.parse_args()

    with open('./config/model_conf.yaml') as f:
        model_conf = yaml.load(f, yaml.FullLoader)

    model_path = 'models'
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    modelDet, cfgDet = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(modelDet, 'cuda:0', cfgDet)

    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    modelAli, cfgAli = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(modelAli, 'cuda:0', cfgAli)

    face_cropper = FaceRecImageCropper()
    to_iterate = list(os.walk(args.input_root))
    
    # Open log file to track multi-face occurrences safely
    with open(os.path.join(args.out_root, 'multi_face_log.txt'), 'w') as log_f:
        for root, dirs, files in tqdm(to_iterate, total=len(to_iterate)):
            for file in files:
                AlignedOneImageUsingFaceXAlignment(args.input_root, args.out_root, os.path.join(root, file), log_f)
