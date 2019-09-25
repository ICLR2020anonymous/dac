import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.emnist import *
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.attention import StackedISAB, PMA, MAB, ISAB, SelfAttentionResUnit
from neural.modules import FixupResUnit, View

from models.base import ModelTemplate

#from data.clustered_dataset import get_random_data_loader, get_saved_data_loader
from data.cluster import sample_anchor
from data.paired_dataset import get_random_data_loader, get_saved_data_loader

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--novel', action='store_true')
parser.add_argument('--num_steps', type=int, default=40000)
parser.add_argument('--testfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=128, num_inds=32):
        super().__init__()

        self.encoder = nn.Sequential(
                FixupResUnit(1, 16, stride=2),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, dim_hids, stride=2),
                nn.AdaptiveAvgPool2d(1),
                View(-1, dim_hids))
        self.fc = nn.Linear(dim_hids, 1)

    def forward(self, x1, x2, mask=None):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return self.fc((f1-f2)**2).squeeze(-1)

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = FindCluster()
        self.metrics = ['bcent']

    @classmethod
    def get_EMNIST(cls, train=True):
        normalize = tvt.Normalize(mean=[0.1751], std=[0.3331])
        if train:
            transform = normalize
            #transform = tvt.Compose(
            #        [tvt.ToPILImage(),
            #         tvt.RandomHorizontalFlip(),
            #         tvt.RandomRotation(90),
            #         tvt.ToTensor(),
            #         normalize])
        else:
            transform = normalize
        return FastEMNIST(os.path.join(datasets_path, 'emnist'),
                split='balanced', download=True, train=train,
                transform=transform)

    def loss_fn(self, dataset, train=True):
        logits = self.net(dataset['X1'].cuda(), dataset['X2'].cuda())
        labels = dataset['labels'].cuda()
        return F.binary_cross_entropy_with_logits(logits, labels)

    def get_train_loader(self):
        return get_random_data_loader(
                self.get_EMNIST(),
                self.batch_size, TRAIN_NUM_PER_CLASS, self.num_steps,
                classes=range(TRAIN_NUM_CLASSES))

    def get_test_loader(self, filename=None):
        filename = self.testfile if filename is None else filename
        classes = range(TRAIN_NUM_CLASSES,
                TRAIN_NUM_CLASSES + TEST_NUM_CLASSES_NOVEL) \
                if self.novel else range(TRAIN_NUM_CLASSES)
        return get_saved_data_loader(
                self.get_EMNIST(False),
                filename, TEST_NUM_PER_CLASS, classes=classes)

def load(args):
    add_args(args, sub_args)

    if args.novel:
        args.testfile = os.path.join(benchmarks_path,
                'emnist_paired_novel.tar' if args.testfile is None else args.testfile)
    else:
        args.testfile = os.path.join(benchmarks_path,
                'emnist_paired.tar' if args.testfile is None else args.testfile)
    return Model(args)
