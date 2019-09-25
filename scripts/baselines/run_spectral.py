import torch
import os
import argparse
from tqdm import tqdm
import time

from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import numpy as np
from scipy.special import logsumexp

from utils.log import get_logger, Accumulator
from utils.paths import benchmarks_path, results_path
from utils.tensor import to_numpy
from utils.plots import scatter, scatter_mog
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--benchmarkfile', type=str, default='mog_10_1000_4.tar')
parser.add_argument('--filename', type=str, default='test.log')

args, _ = parser.parse_known_args()
print(str(args))

benchmark = torch.load(os.path.join(benchmarks_path, args.benchmarkfile))
accm = Accumulator('ari', 'nmi', 'et')
for batch in tqdm(benchmark):
    B = batch['X'].shape[0]
    for b in range(B):
        X = to_numpy(batch['X'][b])
        true_labels = to_numpy(batch['labels'][b].argmax(-1))
        true_K = len(np.unique(true_labels))

        tick = time.time()
        spec = SpectralClustering(n_clusters=true_K,
                affinity='nearest_neighbors',
                n_neighbors=10).fit(X)
        labels = spec.labels_

        accm.update([ARI(true_labels, labels),
            NMI(true_labels, labels, average_method='arithmetic'),
            time.time() - tick])

save_dir = os.path.join(results_path, 'baselines', 'mmaf_spectral')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
logger = get_logger('spectral_baseline', os.path.join(save_dir, args.filename))
logger.info(accm.info())
