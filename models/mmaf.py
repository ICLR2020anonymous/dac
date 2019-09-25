import torch
import torch.nn as nn
import argparse
import os

from utils.paths import benchmarks_path
from utils.misc import add_args
from utils.tensor import meshgrid_around, to_numpy
from utils.plots import scatter
from data.mog import sample_warped_mog

from flows.autoregressive import MAF
from flows.distributions import Normal, FlowDistribution
from neural.attention import StackedISAB, PMA, MAB

from models.base import ModelTemplate

import matplotlib.pyplot as plt

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
    def __init__(self, dim_inputs,
            dim_hids=128, num_inds=32, dim_context=128, num_blocks=4):
        super().__init__()

        self.flow = FlowDistribution(
                MAF(dim_inputs, dim_hids, num_blocks, dim_context=dim_context),
                Normal(dim_inputs, use_context=False))
        self.isab1 = StackedISAB(dim_inputs, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, dim_context)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        H = self.isab1(X, mask=mask)
        Z = self.pma(H, mask=mask)
        context = self.fc1(Z)
        ll = self.flow.log_prob(X, context).unsqueeze(-1)

        H = self.mab(H, Z)
        logits = self.fc2(self.isab2(H, mask=mask))

        return context, ll, logits

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.testfile = os.path.join(benchmarks_path,
                'warped_10_1000_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(benchmarks_path,
                'warped_10_3000_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster(2)

    def sample(self, B, N, K, **kwargs):
        return sample_warped_mog(B, N, K,
                device=torch.device('cuda'), **kwargs)

    def gen_benchmarks(self, force=False):
        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = []
            for _ in range(100):
                bench.append(sample_warped_mog(10, 1000, 4,
                    rand_N=True, rand_K=True))
            torch.save(bench, self.testfile)
        if not os.path.isfile(self.clusterfile) or force:
            print('generating benchmark {}...'.format(self.clusterfile))
            bench = []
            for _ in range(100):
                bench.append(sample_warped_mog(10, 3000, 12,
                    rand_N=True, rand_K=True))
            torch.save(bench, self.clusterfile)

    def plot_clustering(self, X, params, labels):
        B = X.shape[0]
        K = len(params)
        nx, ny = 50, 50
        if B > 1:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                ulabels, colors = scatter(X[b], labels=labels[b], ax=ax)
                for l, c in zip(ulabels, colors):
                    Xbl = X[b][labels[b]==l]
                    Z, x, y = meshgrid_around(Xbl, nx, ny, margin=0.1)
                    ll = self.net.flow.log_prob(Z, context=params[l][b]).reshape(nx, ny)
                    ax.contour(to_numpy(x), to_numpy(y), to_numpy(ll.exp()),
                            zorder=10, alpha=0.3)
        else:
            ulabels, colors = scatter(X[0], labels=labels[0])
            for l, c in zip(ulabels, colors):
                Xbl = X[0][labels[0]==l]
                Z, x, y = meshgrid_around(Xbl, nx, ny, margin=0.1)
                ll = self.net.flow.log_prob(Z, context=params[l][0]).reshape(nx, ny)
                plt.contour(to_numpy(x), to_numpy(y), to_numpy(ll.exp()),
                        zorder=10, alpha=0.3)

    def plot_step(self, X):
        B = X.shape[0]
        self.net.eval()
        params, logits = self.net(X)
        labels = (logits > 0.0).int().squeeze(-1)
        nx, ny = 50, 50
        fig, axes = plt.subplots(2, B//2, figsize=(7*B/5, 5))
        for b, ax in enumerate(axes.flatten()):
            scatter(X[b], labels=labels[b], ax=ax)
            Z, x, y = meshgrid_around(X[b], nx, ny, margin=0.1)
            ll = self.net.flow.log_prob(Z, context=params[b]).reshape(nx, ny)
            ax.contour(to_numpy(x), to_numpy(y), to_numpy(ll.exp()),
                    zorder=10, alpha=0.3)

def load(args):
    add_args(args, sub_args)
    return Model(args)
