import os
import argparse

import torch
import torch.nn.functional as F

from utils.tensor import to_numpy

from models.base import ModelTemplate

from data.clustered_dataset import (ClusteredDataset,
        sample_idxs_and_labels,
        get_random_cluster_loader,
        get_saved_cluster_loader)
from data.cluster import sample_anchors

def min_cluster_loss(logits, labels):
    K = labels.shape[-1]
    bcent = F.binary_cross_entropy_with_logits(
            logits.repeat(1, 1, K), labels, reduction='none').mean(1)
    bcent[labels.sum(1)==0] = float('inf')
    bcent, idx = bcent.min(1)
    bidx = bcent != float('inf')
    return bcent[bidx].mean()

def anchored_cluster_loss(logits, anchor_idxs, labels):
    B = labels.shape[0]
    labels = labels.argmax(-1)
    anchor_labels = labels[torch.arange(B), anchor_idxs]
    targets = (labels == anchor_labels.unsqueeze(-1)).float()
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)

class ImageModelTemplate(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.metrics = ['bcent']

    def get_dataset(self, train=True):
        raise NotImplementedError

    def get_classes(self, train=True):
        return None

    def gen_benchmarks(self, force=False):
        dataset = ClusteredDataset(
                self.get_dataset(False),
                classes=self.get_classes(False))

        def sample_batch(B, N, K):
            batch = sample_idxs_and_labels(dataset.idx_map,
                    B, N, K, rand_N=True, rand_K=True)
            batch['anchor_idxs'] = sample_anchors(B, batch['idxs'].shape[1])
            return batch

        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = [sample_batch(10, 100, 4) for _ in range(100)]
            torch.save(bench, self.testfile)
        if not os.path.isfile(self.clusterfile) or force:
            print('generating benchmark {}...'.format(self.clusterfile))
            bench = [sample_batch(10, 300, 12) for _ in range(100)]
            torch.save(bench, self.clusterfile)

    def loss_fn(self, batch, train=True):
        raise NotImplementedError

    def get_train_loader(self):
        return get_random_cluster_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes())

    def get_test_loader(self, filename=None):
        return get_saved_cluster_loader(self.get_dataset(False),
                (self.testfile if filename is None else filename),
                classes=self.get_classes(False))

    def cluster(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            logits = self.net(X)
            mask = (logits > 0.0)
            done = mask.sum((1,2)) == N
            labels = torch.zeros_like(logits).squeeze(-1).int()
            for i in range(1, max_iter):
                logits = self.net(X, mask=mask)
                ind = logits > 0.0
                labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
                mask[ind] = True

                num_processed = mask.sum((1,2))
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

        fail = done.sum() < B

        if check:
            return None, labels, torch.zeros(1), fail
        else:
            return None, labels, torch.zeros(1)

    # for anchored model
    def cluster_anchored(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            anchor_idxs = sample_anchors(B, N)
            logits = self.net(X, anchor_idxs)
            labels = torch.zeros_like(logits).squeeze(-1).int()
            mask = (logits > 0.0)
            done = mask.sum((1,2)) == N
            for i in range(1, max_iter):
                anchor_idxs = sample_anchors(B, N, mask=mask)
                logits = self.net(X, anchor_idxs, mask=mask)
                ind = logits > 0.0
                labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
                mask[ind] = True

                num_processed = mask.sum((1,2))
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

        fail = done.sum() < B

        if check:
            return None, labels, torch.zeros(1), fail
        else:
            return None, labels, torch.zeros(1)
