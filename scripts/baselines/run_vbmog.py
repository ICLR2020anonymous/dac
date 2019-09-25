import torch
import os
import argparse
from tqdm import tqdm

from scripts.baselines.vbmog import VBMOG
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import numpy as np
from scipy.special import logsumexp

from utils.log import get_logger, Accumulator
from utils.paths import benchmarks_path, results_path
from utils.tensor import to_numpy
from utils.plots import scatter, scatter_mog
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--benchmarkfile', type=str, default='mog_10_1000_4.tar')
parser.add_argument('--k_max', type=int, default=6)
parser.add_argument('--filename', type=str, default='test.log')

args, _ = parser.parse_known_args()
print(str(args))

benchmark = torch.load(os.path.join(benchmarks_path, args.benchmarkfile))

vbmog = VBMOG(args.k_max)
accm = Accumulator('model ll', 'oracle ll', 'ARI', 'NMI', 'k-MAE', 'et')
for dataset in tqdm(benchmark):
    true_labels = to_numpy(dataset['labels'].argmax(-1))
    X = to_numpy(dataset['X'])
    ll = 0
    ari = 0
    nmi = 0
    mae = 0
    et = 0
    for b in range(len(X)):
        tick = time.time()
        vbmog.run(X[b], verbose=False)
        et += time.time() - tick
        ll += vbmog.loglikel(X[b])
        labels = vbmog.labels()
        ari += ARI(true_labels[b], labels)
        nmi += NMI(true_labels[b], labels, average_method='arithmetic')
        mae += abs(len(np.unique(true_labels[b])) - len(np.unique(labels)))

    ll /= len(X)
    ari /= len(X)
    nmi /= len(X)
    mae /= len(X)
    et /= len(X)

    accm.update([ll.item(), dataset['ll'], ari, nmi, mae, et])

save_dir = os.path.join(results_path, 'baselines', 'vbmog')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
logger = get_logger('vbmog_baseline', os.path.join(save_dir, args.filename))
logger.info(accm.info())
