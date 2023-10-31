import numpy as np
import faiss
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--real_root', help='path to the real data (train set)')
parser.add_argument('--test_root', help='path to the test set')
parser.add_argument('--labels_root', help='path to the test labels')
args = parser.parse_args()

train_set = np.load(args.real_root, allow_pickle=True).astype(np.float32)
test_set = np.load(args.test_root, allow_pickle=True).astype(np.float32)
labels = np.load(args.labels_root, allow_pickle=True)

index = faiss.IndexFlatL2(train_set.shape[1])
index.add(train_set)
k_value = 1
D, _ = index.search(test_set, k)
distances = np.sum(D, axis=1)

auc = roc_auc_score(labels, distances)
ap = average_precision_score(labels, distances)
print(f'AP: {ap*100}, AUC: {auc*100}')
