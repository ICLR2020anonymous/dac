import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as tvt

#from data.embedded_tiered_imagenet import EmbeddedTieredImagenet, ROOT
from data.embedded_imagenet import EmbeddedImagenet, ROOT
from data.clustered_dataset import get_random_cluster_loader
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
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=256, num_inds=32):
        super().__init__()
        self.isab = StackedISAB(640, dim_hids, num_inds, 6, p=0.2, ln=True)
        self.fc = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        H_enc = self.isab(X, mask=mask)
        return self.fc(H_enc)

class Model(ImageModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.testfile = os.path.join(benchmarks_path,
                'embedded_imagenet_10_100_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(benchmarks_path,
                'embedded_imagenet_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster()

    def get_dataset(self, train=True):
        transform = tvt.Lambda(lambda x: (x - 0.0038)/0.0118)
        #if train:
        #    transform = tvt.Compose([transform,
        #        tvt.Lambda(lambda x: x + 0.1*torch.randn_like(x))])
        return EmbeddedImagenet(ROOT, train=train, transform=transform)

    def get_train_loader(self):
        return get_random_cluster_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes(), mixup=True)

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
    from utils.paths import results_path
    model = load(sub_args)
    model.net.load_state_dict(torch.load(
        os.path.join(results_path,
        'embedded_imagenet', 'mixup2', 'model.tar')))

    batch = next(iter(model.get_test_loader()))

    logits = model.net(batch['X'])
    print((logits[0]>0).int().squeeze())
