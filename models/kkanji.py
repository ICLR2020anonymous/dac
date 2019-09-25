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

from models.image_base import ImageModelTemplate, min_cluster_loss

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=10000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=128, num_inds=32):
        super().__init__()
        self.encoder = nn.Sequential(
                FixupResUnit(1, 16, stride=2),
                FixupResUnit(16, 32, stride=2),
                FixupResUnit(32, dim_hids, stride=2),
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
        self.net = FindCluster()
        if self.overlap:
            self.testfile = os.path.join(benchmarks_path,
                    'kkanji_overlap_10_100_4.tar' if self.testfile is None else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'kkanji_overlap_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
        else:
            self.testfile = os.path.join(benchmarks_path,
                    'kkanji_10_100_4.tar' if self.testfile is None else self.testfile)
            self.clusterfile = os.path.join(benchmarks_path,
                    'kkanji_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
            self.net = FindCluster()

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

    def loss_fn(self, batch, train=True):
        X = batch['X'].cuda()
        logits = self.net(X)
        labels = batch['labels'].cuda().float()
        return min_cluster_loss(logits, labels)

def load(args):
    add_args(args, sub_args)
    return Model(args)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    from utils.paths import results_path
    model = load(sub_args)
    model.net.cuda()
    model.net.load_state_dict(
            torch.load(
                os.path.join(results_path, 'kkanji', 'trial', 'model.tar')))

    batch = next(iter(model.get_test_loader()))

    logits = model.net(batch['X'].cuda())

    print((logits[0] > 0).int().squeeze())
    print(logits[0].squeeze())
