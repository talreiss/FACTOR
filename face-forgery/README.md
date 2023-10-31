## 1. Installation
First, activate the virtual environment:
```
source ../venv/bin/activate
```

Then, clone the FaceX-Zoo [repository](https://github.com/JDAI-CV/FaceX-Zoo.git):
```
git clone https://github.com/JDAI-CV/FaceX-Zoo.git
```

Now, please do the following commands:
```
mv extract_frames.py FaceX-Zoo/face_sdk/
mv detect_and_align.py FaceX-Zoo/face_sdk/
mv extract_feature.py FaceX-Zoo/test_protocol/
mv eval.py FaceX-Zoo/test_protocol/
mv backbone_conf.yaml FaceX-Zoo/test_protocol/
```

FaceX-Zoo checkpoints are available in the following [link](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode). Here is the model we used:
```
https://drive.google.com/drive/folders/1h_meJetsaVUm-37Wqo-o3ed9lyWcS8-B
```
Download and place `Epoch_17.pt` in `./FaceX-Zoo/test_protocol/`.

##  2. Usage
### 2.1  Extract frames from `.mp4` files
```
cd FaceX-Zoo/face_sdk/
python extract_frames.py [--input_root] [--out_root] [--num_frames]
```
Where `--input_root` is the directory containing the `.mp4` files, `--out_root` is the output directory, and `--num_frames` is the number of frames to extract.  

### 2.2 Detect and align

For extracting ROIs and align them, run the following command:
```
cd FaceX-Zoo/face_sdk/
python detect_and_align.py [--input_root] [--out_root]
```
Where `--input_root` is the directory containing the extracted frames, and `--out_root` is the output directory.  


### 2.3 Feature extraction
For extracting face representations, run the following command:
```
cd FaceX-Zoo/test_protocol/
python extract_feature.py [--input_root]
```
Where `--input_root` is the directory containing the extracted frames after detection and alignment.

### 2.4 Evaluation
The evaluation stage assumes that the user has already obtained:
1. `.npy` file containing the extracted features of a claimed identity reference set (e.g. Barack Obama images).
2. `.npy` file containing the extracted features of a test set containing the observed identity and a claimed identity (e.g. images of attacks on Obama and authentic Obama images).
3. `.npy` file containing the labels of each sample in the test set.

Finally, you can evaluate by running the following command:
```
cd FaceX-Zoo/test_protocol/
python eval.py [--real_root] [--test_root] [--labels_root]
```
Where `--real_root` is the path to reference set `.npy` file, `--test_root` is the path to test set `.npy` file, and `--labels_root` is the path to test corresponding labels `.npy` file.

## FaceX-Zoo
The implementation we offer is based upon FaceX-Zoo and its implementation. If you have any implementation-related questions, please refer to the original FaceX-Zoo [repository](https://github.com/JDAI-CV/FaceX-Zoo.git). 