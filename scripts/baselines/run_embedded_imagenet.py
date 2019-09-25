import os
import argparse
import time

import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from utils.log import Accumulator, get_logger
from utils.paths import benchmarks_path
from utils.tensor import to_numpy

from data.embedded_imagenet import EmbeddedImagenet, ROOT
from data.clustered_dataset import get_saved_cluster_loader
import torchvision.transforms as tvt

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default='kmeans')
parser.add_argument('--filename', type=str, default='test.log')
args = parser.parse_args()

transform = tvt.Lambda(lambda x: (x - 0.0038)/0.0118)
dataset = EmbeddedImagenet(ROOT, train=False, transform=transform)
filename = os.path.join(benchmarks_path, 'embedded_imagenet_10_300_12.tar')
loader = get_saved_cluster_loader(dataset, filename)
accm = Accumulator('ari', 'nmi', 'et')

for batch in tqdm(loader):
    B = batch['X'].shape[0]
    for b in range(B):
        X = to_numpy(batch['X'][b])
        true_labels = to_numpy(batch['labels'][b].argmax(-1))
        true_K = len(np.unique(true_labels))

        tick = time.time()

        # KMeans
        if args.alg == 'kmeans':
            kmeans = KMeans(n_clusters=true_K).fit(X)
            labels = kmeans.labels_

        # Spectral
        if args.alg == 'spectral':
            spec = SpectralClustering(n_clusters=true_K,
                    affinity='nearest_neighbors',
                    n_neighbors=10,
                    ).fit(X)
            labels = spec.labels_

        accm.update([ARI(true_labels, labels),
            NMI(true_labels, labels, average_method='arithmetic'),
            time.time() - tick])

logger = get_logger(args.alg, args.filename)
logger.info(accm.info())
