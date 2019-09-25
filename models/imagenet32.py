import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.imagenet32 import Imagenet32, ROOT, MEAN, STD
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.attention import StackedISAB, PMA, MAB, ISAB
from neural.modules import FixupResUnit, View

from models.image_base import ImageModelTemplate, min_cluster_loss

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=256, num_inds=32):
        super().__init__()
        self.encoder = nn.Sequential(
                FixupResUnit(3, 16, stride=2),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, 64, stride=2),
                FixupResUnit(64, dim_hids),
                nn.AdaptiveAvgPool2d(1))
        self.isab = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        B, N, C, H, W = X.shape
        X_flat = X.view(B*N, C, H, W)
        H_enc = self.encoder(X_flat).view(B, N, -1)
        H_enc = self.isab(H_enc, mask=mask)
        return self.fc(H_enc)

class Model(ImageModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        if self.overlap:
            self.testfile = os.path.join(benchmarks_path,
                    'imagenet32_overlap_10_100_4.tar' if self.testfile is none else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'imagenet32_overlap_10_300_12.tar' if self.clusterfile is none else self.clusterfile)
        else:
            self.testfile = os.path.join(benchmarks_path,
                    'imagenet32_10_100_4.tar' if self.testfile is none else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'imagenet32_10_300_12.tar' if self.clusterfile is none else self.clusterfile)
        self.net = FindCluster()

    def get_dataset(self, train=True):
        transforms = [tvt.ToTensor(),
                tvt.Normalize(mean=MEAN, std=STD)]
        if train:
            transforms = [tvt.RandomCrop(32, padding=4),
                    tvt.RandomHorizontalFlip()] + transforms
        transform = tvt.Compose(transforms)
        return Imagenet32(ROOT, train=train, transform=transform)

    def get_classes(self, train=True):
        if train:
            return range(800)
        else:
            if self.overlap:
                return range(800)
            else:
                return range(800, 1000)

    def loss_fn(self, batch, train=True):
        X = batch['X'].cuda()
        logits = self.net(X)
        labels = batch['labels'].cuda().float()
        return min_cluster_loss(logits, labels)

def load(args):
    add_args(args, sub_args)
    return Model(args)
