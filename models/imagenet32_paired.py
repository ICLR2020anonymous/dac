import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from data.imagenet32 import Imagenet32, ROOT, MEAN, STD
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path, results_path

from neural.attention import StackedISAB, PMA, MAB, ISAB
from neural.modules import FixupResUnit, View

from models.image_pair_base import ImagePairModelTemplate

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=200000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--ckpt', type=str, default='imagenet32_ptr/trial')

sub_args, _ = parser.parse_known_args()

class ClassifyPair(nn.Module):
    def __init__(self, dim_hids=256):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=7, padding=3),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, 32),
                FixupResUnit(32, 64, stride=2),
                FixupResUnit(64, 64),
                FixupResUnit(64, 128, stride=2),
                FixupResUnit(128, dim_hids),
                nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
                View(-1, dim_hids),
                nn.Linear(dim_hids, 1))

    def forward(self, x1, x2):
        with torch.no_grad():
            f1 = self.encoder(x1)
            f2 = self.encoder(x2)
        return self.fc((f1-f2)**2).squeeze(-1)
        #return self.fc((self.encoder(x1) - self.encoder(x2))**2).squeeze(-1)

class Model(ImagePairModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = ClassifyPair()

        if self.overlap:
            self.testfile = os.path.join(benchmarks_path,
                    'imagenet32_overlap_pairs.tar' if self.testfile is None else self.testfile)
        else:
            self.testfile = os.path.join(benchmarks_path,
                    'imagenet32_pairs.tar' if self.testfile is None else self.testfile)

    def load_from_ckpt(self):
        if self.ckpt is not None:
            ckpt = torch.load(os.path.join(results_path, self.ckpt, 'model.tar'))
            self.net.load_state_dict(ckpt, strict=False)

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

def load(args):
    add_args(args, sub_args)
    return Model(args)
