## 1. Installation
First, activate the virtual environment:
```
source ../venv/bin/activate
```
Then:
```
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../../
```

Now, please do the following commands:
```
mv preprocess.py av_hubert/avhubert/
mv inference.py av_hubert/avhubert/
mv eval.py av_hubert/avhubert/
```

Moreover, as skvideo and fairseq is deprecated you should use the following command:
```
mv abstract.py ../venv/lib/python3.9/site-packages/skvideo/io/
mv indexed_dataset.py av_hubert/fairseq/fairseq/data/
```

Install necessary tools for preprocessing:
```
mkdir -p misc/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O misc/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d misc/shape_predictor_68_face_landmarks.dat.bz2
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O misc/20words_mean_face.npy
```

AV-HuBERT checkpoints are available in the following [link](https://facebookresearch.github.io/av_hubert/). Here is the model we used:
```
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/large_lrs3_30h.pt -O misc/model.pt
```

##  2. Usage
### 2.1  Data download and Preparation
Download the FakeAVCeleb dataset from:
```
https://sites.google.com/view/fakeavcelebdash-lab/download
```
Extract its contents into the `data` folder.

### 2.2 Pre-processing

For extracting ROIs and wav files, run the following command:
```
cd av_hubert/avhubert
python preprocess.py [--category]
```
Where `--category` is one of the following: `['rvra', 'rvfa', 'fvra', 'fvfa'']`

### 2.3 Feature extraction
For extracting audio and video representations, run the following command:
```
cd av_hubert/avhubert
python inference.py 0
```

### 2.4 Evaluation
Finally, you can evaluate by running the following command:
```
cd av_hubert/avhubert
python eval.py [--category]
```
Where `--category` is one of the following: `['rvfa', 'fvra', 'fvfa-wl', 'fvfa-fs', 'fvfa-gan']`. \
When the `category` is `'rvfa'`, RVRA will be compared with RVFA.

## AV-HuBERT

The implementation we offer is based upon AV-HuBERT and its implementation. If you have any implementation-related questions, please refer to the original AV-HuBERT [repository](https://github.com/facebookresearch/av_hubert.git). 