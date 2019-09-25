import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.cluster import sample_anchors
from data.embedded_imagenet import EmbeddedImagenet, ROOT
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.attention import StackedISAB, PMA, MAB, ISAB
from neural.modules import FixupResUnit, View

from models.image_base import anchored_cluster_loss
from models.embedded_imagenet import Model

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
#parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=256, num_inds=32):
        super().__init__()
        self.isab = StackedISAB(640, dim_hids, num_inds, 6, p=0.2, ln=True)
        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.fc = nn.Linear(dim_hids, 1)

    def forward(self, X, anchor_idxs, mask=None):
        H_enc = self.isab(X, mask=mask)
        anchors = H_enc[torch.arange(X.shape[0]), anchor_idxs].unsqueeze(1)
        H_enc = self.mab(H_enc, anchors)
        return self.fc(H_enc)

class Model(Model):
    def __init__(self, args):
        super().__init__(args)
        self.net = FindCluster()

    def loss_fn(self, batch, train=True):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda().float()
        anchor_idxs = sample_anchors(X.shape[0], X.shape[1]).cuda() \
                if train else batch['anchor_idxs'].cuda()
        logits = self.net(X, anchor_idxs)
        return anchored_cluster_loss(logits, anchor_idxs, labels)

    def cluster(self, *args, **kwargs):
        return super().cluster_anchored(*args, **kwargs)

def load(args):
    add_args(args, sub_args)
    return Model(args)
