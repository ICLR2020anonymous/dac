import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as WN
import torchvision.transforms as tvt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data.emnist import FastEMNIST
from data.clustered_dataset import (ClusteredDataset,
        sample_idxs_and_labels,
        get_random_cluster_loader,
        get_saved_cluster_loader)

from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path
from utils.tensor import to_numpy

from neural.attention import StackedISAB, PMA, MAB, ISAB
from neural.modules import FixupResUnit, View

from flows.autoregressive import MAF
from flows.distributions import FlowDistribution, Normal, Bernoulli

from models.base import ModelTemplate

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=40000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

# for visualization
parser.add_argument('--vB', type=int, default=1)
parser.add_argument('--vN', type=int, default=100)
parser.add_argument('--vK', type=int, default=4)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_lats, dim_hids=128, num_inds=32):
        super().__init__()

        self.encoder = nn.Sequential(
                View(-1, 784),
                WN(nn.Linear(784, dim_hids)),
                nn.ELU(),
                WN(nn.Linear(dim_hids, dim_hids)),
                nn.ELU(),
                WN(nn.Linear(dim_hids, dim_hids)))

        self.isab1 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, dim_hids)

        self.posterior = Normal(dim_lats, use_context=True,
                context_enc=nn.Linear(2*dim_hids, 2*dim_lats))
        self.prior = FlowDistribution(
                MAF(dim_lats, dim_hids, 4, dim_context=dim_hids, inv_linear=True),
                Normal(dim_lats))

        self.decoder = nn.Sequential(
                WN(nn.Linear(dim_lats + dim_hids, dim_hids)),
                nn.ELU(),
                WN(nn.Linear(dim_hids, dim_hids)),
                nn.ELU(),
                WN(nn.Linear(dim_hids, 784)),
                View(-1, 1, 28, 28))
        self.likel = Bernoulli((1, 28, 28), use_context=True)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        B, N, C, H, W = X.shape
        x = X.view(B*N, C, H, W)
        h_enc = self.encoder(x)
        H_enc = self.isab1(h_enc.view(B, N, -1), mask=mask)
        Z = self.pma(H_enc, mask=mask)
        context = self.fc1(Z).repeat(1, N, 1).view(B*N, -1)

        if self.training:
            z, log_q = self.posterior.sample(context=torch.cat([h_enc, context], -1))
        else:
            z, log_q = self.posterior.mean(context=torch.cat([h_enc, context], -1))
        log_p = self.prior.log_prob(z, context=context)
        kld = (log_q - log_p).view(B, N, -1)

        h_dec = self.decoder(torch.cat([z, context], -1))
        ll = self.likel.log_prob(x, context=h_dec).view(B, N, -1) - kld
        ll /= C*H*W

        H_dec = self.mab(H_enc, Z)
        logits = self.fc2(self.isab2(H_dec, mask=mask))

        return context, ll, logits

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        if self.overlap:
            self.testfile = os.path.join(benchmarks_path,
                    'emnist_overlap_10_100_4.tar' if self.testfile is None else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'emnist_overlap_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
        else:
            self.testfile = os.path.join(benchmarks_path,
                    'emnist_10_100_4.tar' if self.testfile is None else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'emnist_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster(64)

    def get_classes(self, train=True):
        if train:
            return range(30)
        else:
            if self.overlap:
                return range(30)
            else:
                return range(30, 47)

    def get_dataset(self, train=True):
        return FastEMNIST(os.path.join(datasets_path, 'emnist'),
                split='balanced', download=True, train=train,
                transform=tvt.Lambda(torch.bernoulli))

    def gen_benchmarks(self, force=False):
        dataset = ClusteredDataset(self.get_dataset(False),
                classes=self.get_classes(False))
        def sample_batch(B, N, K):
            batch = sample_idxs_and_labels(dataset.idx_map,
                    B, N, K, rand_N=True, rand_K=True)
            return batch

        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = [sample_batch(10, 100, 4) for _ in range(100)]
            torch.save(bench, self.testfile)
        if not os.path.isfile(self.clusterfile) or force:
            print('generating benchmark {}...'.format(self.clusterfile))
            bench = [sample_batch(10, 300, 12) for _ in range(100)]
            torch.save(bench, self.clusterfile)

    def sample(self, B, N, K, **kwargs):
        loader = get_random_cluster_loader(self.get_dataset(False),
                B, N, K, self.num_steps,
                classes=self.get_classes(False), **kwargs)
        return next(iter(loader))

    def get_train_loader(self):
        return get_random_cluster_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes(True))

    def get_test_loader(self, filename=None):
        return get_saved_cluster_loader(self.get_dataset(False),
                (self.testfile if filename is None else filename),
                classes=self.get_classes(False))

    def plot_step(self, X):
        if X.shape[0] > 1:
            raise ValueError('No support for visualization when B > 1')

        self.net.eval()
        with torch.no_grad():
            X = X.cuda()
            params, ll, logits = self.net(X)
            labels = (logits > 0.0).squeeze()
            print(labels.shape)

            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            img = make_grid(X[0][labels==1])
            axes[0].imshow(to_numpy(img).transpose(1,2,0))
            axes[0].axis('off')
            axes[0].set_title('In cluster')

            z, _ = self.net.prior.sample(num_samples=X.shape[1],
                    context=params, device=X.device)
            h = self.net.decoder(torch.cat([z, params], -1))
            x_gen, _ = self.net.likel.sample(context=h)
            img = make_grid(x_gen)
            axes[1].imshow(to_numpy(img).transpose(1,2,0))
            axes[1].axis('off')
            axes[1].set_title('Generated')

            img = make_grid(X[0][labels==0])
            axes[2].imshow(to_numpy(img).transpose(1,2,0))
            axes[2].axis('off')
            axes[2].set_title('Not in cluster')

            plt.tight_layout()

    def plot_clustering(self, X, params, labels):
        if X.shape[0] > 1:
            raise ValueError('No support for visualization when B > 1')

        X = X[0]
        labels = labels[0]
        unique_labels = torch.unique(labels)

        fig, axes = plt.subplots(len(unique_labels), 1, figsize=(20, 20))
        for i, l in enumerate(unique_labels):
            Xl = X[labels==l]
            img = make_grid(Xl, nrow=10)
            axes[i].imshow(to_numpy(img).transpose(1,2,0))
            axes[i].axis('off')
        plt.tight_layout()

def load(args):
    add_args(args, sub_args)
    return Model(args)
