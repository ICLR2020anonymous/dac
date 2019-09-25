import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.kkanji import KKanji
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.attention import StackedISAB, PMA, MAB, ISAB
from neural.modules import FixupResUnit, View

from models.image_pair_base import ImagePairModelTemplate

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=40000)
parser.add_argument('--testfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class ClassifyPair(nn.Module):
    def __init__(self, dim_hids=128):
        super().__init__()
        self.encoder = nn.Sequential(
                FixupResUnit(1, 16, stride=2),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, dim_hids, stride=2),
                nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
                View(-1, dim_hids),
                nn.Linear(dim_hids, 1))

    def forward(self, x1, x2):
        return self.fc((self.encoder(x1) - self.encoder(x2))**2).squeeze(-1)

class Model(ImagePairModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = ClassifyPair()

        if self.overlap:
            self.testfile = os.path.join(benchmarks_path,
                    'kkanji_overlap_pairs.tar' if self.testfile is None else self.testfile)
        else:
            self.testfile = os.path.join(benchmarks_path,
                    'kkanji_pairs.tar' if self.testfile is None else self.testfile)

    def get_dataset(self, train=True):
        transform = tvt.Normalize(mean=[0.2170], std=[0.3787])
        if train:
            transform = tvt.Compose([
                tvt.ToPILImage(),
                tvt.RandomHorizontalFlip(),
                tvt.RandomCrop(28),
                tvt.ToTensor(),
                transform])

        return KKanji(os.path.join(datasets_path, 'kkanji'),
                train=train, transform=transform)

    def get_classes(self, train=True):
        if train:
            return range(700)
        else:
            if self.overlap:
                return range(700)
            else:
                return range(700, 813)

def load(args):
    add_args(args, sub_args)
    return Model(args)
