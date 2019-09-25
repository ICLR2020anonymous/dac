import os
import argparse

import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI

from utils.log import Accumulator
from utils.paths import benchmarks_path, datasets_path
from utils.tensor import to_numpy

from data.kkanji import KKanji
from data.clustered_dataset import get_saved_cluster_loader
import torchvision.transforms as tvt

transform = tvt.Normalize(mean=[0.2170], std=[0.3787])
dataset = KKanji(os.path.join(datasets_path, 'kkanji'), train=False, transform=transform)
filename = os.path.join(benchmarks_path, 'kkanji_10_300_12.tar')
loader = get_saved_cluster_loader(dataset, filename, classes=range(700, 813))
accm = Accumulator('ari', 'k-mae')

for batch in tqdm(loader):
    B = batch['X'].shape[0]
    for b in range(B):
        X = to_numpy(batch['X'][b]).reshape(-1, 784)
        true_labels = to_numpy(batch['labels'][b].argmax(-1))
        true_K = len(np.unique(true_labels))

        # KMeans
        kmeans = KMeans(n_clusters=true_K).fit(X)
        labels = kmeans.labels_

        # Spectral
        #spec = SpectralClustering(n_clusters=true_K).fit(X)
        #labels = spec.labels_

        #gmm = GaussianMixture(n_components=true_K).fit(X)
        #labels = gmm.predict(X)

        accm.update([ARI(true_labels, labels), abs(len(np.unique(labels))-true_K)])

print(accm.info())
