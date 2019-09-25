import argparse
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import benchmarks_path
from utils.misc import add_args
from utils.plots import scatter, draw_ellipse, scatter_mog

from data.mvn import MultivariateNormalDiag
from data.mog import sample_mog

from neural.attention import StackedISAB, PMA, MAB

from models.base import ModelTemplate

parser = argparse.ArgumentParser()

# for training
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

# for visualization
parser.add_argument('--vB', type=int, default=10)
parser.add_argument('--vN', type=int, default=1000)
parser.add_argument('--vK', type=int, default=4)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, mvn, dim_hids=128, num_inds=32):
        super().__init__()
        self.mvn = mvn

        self.isab1 = StackedISAB(mvn.dim, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, mvn.dim_params)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        H_enc = self.isab1(X, mask=mask)
        Z = self.pma(H_enc, mask=mask)
        params = self.mvn.transform(self.fc1(Z))
        ll = self.mvn.log_prob(X, params)

        H_dec = self.mab(H_enc, Z)
        logits = self.fc2(self.isab2(H_dec, mask=mask))

        return params, ll, logits

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.testfile = os.path.join(benchmarks_path,
                'mog_10_1000_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(benchmarks_path,
                'mog_10_3000_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster(MultivariateNormalDiag(2))

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, 1000, 4,
                    rand_N=True, rand_K=True, return_ll=True))
            torch.save(bench, self.testfile)
        if not os.path.isfile(self.clusterfile) or force:
            print('generating benchmark {}...'.format(self.clusterfile))
            bench = []
            for _ in range(100):
                bench.append(sample_mog(10, 3000, 12,
                    rand_N=True, rand_K=True, return_ll=True))
            torch.save(bench, self.clusterfile)

    def sample(self, B, N, K, **kwargs):
        return sample_mog(B, N, K, device=torch.device('cuda'), **kwargs)

    def plot_clustering(self, X, params, labels):
        B = X.shape[0]
        mu, cov = self.net.mvn.stats(torch.cat(params, 1))
        if B == 1:
            scatter_mog(X[0], labels[0], mu[0], cov[0])
        else:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                scatter_mog(X[b], labels[b], mu[b], cov[b], ax=ax)

    def plot_step(self, X):
        B = X.shape[0]
        self.net.eval()
        params, _, logits = self.net(X)
        mu, cov = self.net.mvn.stats(params)
        labels = (logits > 0.0).int().squeeze(-1)
        if B == 1:
            scatter(X[0], labels=labels[0])
            draw_ellipse(mu[0][0], cov[0][0])
        else:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                scatter(X[b], labels=labels[b], ax=ax)
                draw_ellipse(mu[b][0], cov[b][0], ax=ax)

def load(args):
    add_args(args, sub_args)
    return Model(args)
