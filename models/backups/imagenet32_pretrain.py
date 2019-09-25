import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt

from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.attention import StackedISAB, PMA, MAB
from neural.modules import FixupResUnit, View

from models.base import ModelTemplate

from data.clustered_dataset import get_random_data_loader, get_saved_data_loader
from data.imagenet32 import *

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--num_steps', type=int, default=80000)

sub_args, _ = parser.parse_known_args()

class Classifier(nn.Module):
    def __init__(self, dim_hids=256):
        super().__init__()

        self.encoder = nn.Sequential(
                ConvResUnit(3, 16, stride=2),
                ConvResUnit(16, 32, stride=2),
                ConvResUnit(32, 64, stride=2),
                ConvResUnit(64, dim_hids),
                nn.AdaptiveAvgPool2d(1))
        self.cls = nn.Linear(dim_hids, TRAIN_NUM_CLASSES)

    def forward(self, x):
        return self.cls(self.encoder(x))

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = Classifier()
        self.metrics = ['top1-acc', 'top5-acc']

    def get_ImageNet32(self, train=True):
        transforms = [tvt.ToTensor(),
                tvt.Normalize(mean=MEAN, std=STD)]
        if train:
            transforms = [tvt.RandomCrop(32),
                    tvt.RandomHorizontalFlip()] + transforms
        transform = tvt.Compose(transforms)
        return ImageNet32(ROOT, train=train, transform=transform)

    def get_train_loader(self):
        dataset = self.get_ImageNet32()
        loader = DataLoader(dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=4)
        num_epochs = self.num_steps // len(loader)
        for _ in range(num_epochs):
            for x, y in loader:
                yield x, y
