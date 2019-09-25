import os
import argparse
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.emnist import *
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path, results_path

from neural.attention import StackedISAB, PMA, MAB, ISAB, SelfAttentionResUnit
from neural.modules import ConvResUnit, FixupResUnit, View

from models.base import ModelTemplate

from data.clustered_dataset import get_random_data_loader, get_saved_data_loader
from data.cluster import sample_anchor

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--phase', type=int, default=1)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, phase=1, dim_hids=128, num_inds=32):
        super().__init__()
        self.phase = phase
        self.encoder = nn.Sequential(
                FixupResUnit(1, 16, stride=2),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, dim_hids, stride=2),
                nn.AdaptiveAvgPool2d(1))
        self.isab = StackedISAB(dim_hids, dim_hids, num_inds, 4, p=0.2)
        self.fc = nn.Linear(dim_hids, 1)

    def parameters(self):
        #modules = ['encoder']
        #if self.phase == 1:
        #    modules += ['isab', 'fc']
        modules = ['isab', 'fc']
        if self.phase == 1:
            modules += ['encoder']
        return chain(*[getattr(self, m).parameters() for m in modules])

    def forward(self, X, mask=None):
        B, N, C, H, W = X.shape
        X_flat = X.contiguous().view(B*N, C, H, W)
        H_enc = self.encoder(X_flat).view(B, N, -1)
        H_enc = self.isab(H_enc, mask=mask)
        return self.fc(H_enc)

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = FindCluster(self.phase)
        self.metrics = ['bcent']

    def load_from_ckpt(self):
        if self.ckpt is None:
            return
        modelfile = os.path.join(results_path, self.ckpt, 'model.tar')
        self.net.load_state_dict(torch.load(modelfile))

    @classmethod
    def get_EMNIST(cls, train=True):
        normalize = tvt.Normalize(mean=[0.1751], std=[0.3331])
        if train:
            transform = normalize
        else:
            transform = normalize
        return FastEMNIST(os.path.join(datasets_path, 'emnist'),
                split='balanced', download=True, train=train,
                transform=transform)

    def loss_fn(self, dataset, train=True):
        X = dataset['X'].cuda()
        logits = self.net(X)
        labels = dataset['labels'].cuda().float()
        K = labels.shape[-1]
        bcent = F.binary_cross_entropy_with_logits(
                logits.repeat(1, 1, K), labels, reduction='none').mean(1)
        bcent[labels.sum(1)==0] = float('inf')
        bcent, idx = bcent.min(1)
        bidx = bcent != float('inf')
        return bcent[bidx].mean()

    def get_train_loader(self):
        if self.phase == 1:
            classes = range(TRAIN_NUM_CLASSES//2)
        else:
            classes = range(TRAIN_NUM_CLASSES)
        return get_random_data_loader(
                self.get_EMNIST(),
                self.B, self.N, self.K, self.num_steps,
                TRAIN_NUM_PER_CLASS, classes=classes)

    def get_test_loader(self, filename=None):
        filename = self.testfile if filename is None else filename
        if self.phase == 1:
            classes = range(TRAIN_NUM_CLASSES//2)
        else:
            classes = range(TRAIN_NUM_CLASSES, TRAIN_NUM_CLASSES + TEST_NUM_CLASSES_NOVEL)
        return get_saved_data_loader(
                self.get_EMNIST(False),
                filename, TEST_NUM_PER_CLASS, classes=classes)

def load(args):
    add_args(args, sub_args)

    if args.testfile is None:
        args.testfile = os.path.join(os.path.join(benchmarks_path),
                'emnist_phase{}_10_1000_4.tar'.format(args.phase))

    return Model(args)
