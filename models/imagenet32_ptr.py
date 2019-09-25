import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tvt
from torch.utils.data import DataLoader

from data.imagenet32 import Imagenet32, ROOT, MEAN, STD
from utils.misc import add_args
from utils.paths import datasets_path, benchmarks_path

from neural.modules import FixupResUnit, View

from models.base import ModelTemplate

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--lr', type=int, default=1e-1)

sub_args, _ = parser.parse_known_args()

class Classifier(nn.Module):
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
        self.classifier = nn.Sequential(
                View(-1, dim_hids),
                nn.Linear(dim_hids, 800))

    def forward(self, x):
        return self.classifier(self.encoder(x))

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = Classifier()
        self.metrics = ['cent', 'acc']
        self.num_steps = self.num_epochs * (717142 // self.batch_size)

    def get_dataset(self, train=True):
        transforms = [tvt.ToTensor(),
                tvt.Normalize(mean=MEAN, std=STD)]
        if train:
            transforms = [tvt.RandomCrop(32, padding=4),
                    tvt.RandomHorizontalFlip()] + transforms
        transform = tvt.Compose(transforms)
        return Imagenet32(ROOT, train=train, transform=transform,
                classes=range(800))

    def build_optimizer(self):
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.num_steps)
        return optimizer, scheduler

    def get_train_loader(self):
        loader = DataLoader(self.get_dataset(),
                batch_size=self.batch_size,
                num_workers=4, shuffle=True)
        for _ in range(self.num_epochs):
            for x, y in loader:
                yield x, y

    def get_test_loader(self):
        return DataLoader(self.get_dataset(False),
                batch_size=self.batch_size, num_workers=4)

    def loss_fn(self, batch, train=True):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        logits = self.net(x)
        cent = F.cross_entropy(logits, y)
        if train:
            return cent
        else:
            return cent, (logits.argmax(-1)==y).float().mean()

def load(args):
    add_args(args, sub_args)
    return Model(args)
