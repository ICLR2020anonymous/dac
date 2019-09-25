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

from neural.attention import StackedISAB, aPMA, StackedSAB

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

def v_to_c(v_logits):
    log_v = (torch.sigmoid(v_logits) + 1e-10).log()
    log_c = log_v.cumsum(-1)
    c_logits = log_c - (1 - log_c.exp() + 1e-10).log()
    return c_logits

class ACTSetTransformer(nn.Module):
    def __init__(self, mvn, dim_hids=128, num_inds=32):
        super().__init__()
        self.mvn = mvn

        self.isab = StackedISAB(mvn.dim, dim_hids, num_inds, 4)
        self.apma = aPMA(dim_hids, dim_hids)
        self.sab = StackedSAB(dim_hids, dim_hids, 2)
        self.fc = nn.Linear(dim_hids, 2 + mvn.dim_params)

    def forward(self, X, K_max, K_true):
        B = X.shape[0]
        device = X.device
        H_enc = self.isab(X)

        v_logits = torch.zeros(B, K_max, device=device)
        c_targets = torch.ones(B, K_max, device=device)
        for b in range(B):
            c_targets[b, K_true[b]-1:] = 0.0

        pi_logits = torch.Tensor(B, K_max).fill_(-1e+10).to(device)
        params = torch.ones(B, K_max, self.mvn.dim_params, device=device)

        if self.training:
            for k in range(K_max):
                outs = self.fc(self.sab(self.apma(H_enc, k+1)))
                v_logits[:,k] = outs[...,0].mean(-1)
                for b in range(B):
                    if k+1 == K_true[b]:
                        pi_logits[b, :k+1] = outs[b,:,1]
                        params[b,:k+1] = self.mvn.transform(outs[b,:,2:])
            c_logits = v_to_c(v_logits)

        else:
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for k in range(K_max):
                outs = self.fc(self.sab(self.apma(H_enc, k+1)))
                v_logits[:,k] = outs[...,0].mean(-1)
                c_logits = v_to_c(v_logits[...,:k+1])
                for b in range(B):
                    if not done[b] and c_logits[b,k] < 0:
                        pi_logits[b, :k+1] = outs[b,:,1]
                        params[b, :k+1] = self.mvn.transform(outs[b,:,2:])
                        done[b] = True

                if done.int().sum() == B:
                    break

            for b in range(B):
                if not done[b]:
                    pi_logits[b, :k+1] = outs[b,:,1]
                    params[b, :k+1] = self.mvn.transform(outs[b,:,2:])
            c_logits = v_to_c(v_logits)

        pi = torch.softmax(pi_logits, -1) + 1e-10
        ll = self.mvn.log_prob(X, params) + pi.unsqueeze(1).log()
        ll = ll.logsumexp(-1).mean()
        bcent = F.binary_cross_entropy_with_logits(c_logits, c_targets)

        return params, ll, bcent

    def cluster(self, X, K_max):
        B = X.shape[0]
        device = X.device
        H_enc = self.isab(X)

        v_logits = torch.zeros(B, K_max, device=device)
        pi_logits = torch.Tensor(B, K_max).fill_(-1e+10).to(device)
        params = torch.ones(B, K_max, self.mvn.dim_params, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        fail = True
        for k in range(K_max):
            outs = self.fc(self.sab(self.apma(H_enc, k+1)))
            v_logits[:,k] = outs[...,0].mean(-1)
            c_logits = v_to_c(v_logits[...,:k+1])
            for b in range(B):
                if not done[b] and c_logits[b, k] < 0:
                    pi_logits[b, :k+1] = outs[b,:,1]
                    params[b, :k+1] = self.mvn.transform(outs[b,:,2:])
                    done[b] = True

                if done.int().sum() == B:
                    fail = False
                    break

        for b in range(B):
            if not done[b]:
                pi_logits[b, :k+1] = outs[b,:,1]
                params[b, :k+1] = self.mvn.transform(outs_b[b,:,2:])
                done[b] = True

        pi = torch.softmax(pi_logits, -1) + 1e-10
        ll = self.mvn.log_prob(X, params) + pi.unsqueeze(1).log()
        labels = ll.argmax(-1)
        ll = ll.logsumexp(-1).mean()
        return params, labels, ll, fail

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.testfile = os.path.join(benchmarks_path,
                'mog_10_1000_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(benchmarks_path,
                'mog_10_3000_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = ACTSetTransformer(MultivariateNormalDiag(2))

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

    def loss_fn(self, batch, train=True):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda()
        K_max = labels.shape[2]
        K_true = (labels.sum(1) > 0).sum(-1)

        _, ll, bcent = self.net(X, K_max, K_true)
        if train:
            return bcent - ll
        else:
            return ll, bcent

    def cluster(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            params, labels, ll, fail = self.net.cluster(X, max_iter)

        if check:
            return params, labels, ll, fail
        else:
            return params, labels, ll

    def plot_clustering(self, X, params, labels):
        B = X.shape[0]
        mu, cov = self.net.mvn.stats(params)
        if B == 1:
            scatter_mog(X[0], labels[0], mu[0], cov[0])
        else:
            fig, axes = plt.subplots(2, B//2, figsize=(2.5*B, 10))
            for b, ax in enumerate(axes.flatten()):
                scatter_mog(X[b], labels[b], mu[b], cov[b], ax=ax)

def load(args):
    add_args(args, sub_args)
    return Model(args)
