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

from flows.autoregressive import MAF
from flows.distributions import Normal, FlowDistribution

from models.base import compute_filter_loss
from models.image_base import ImageModelTemplate, min_cluster_loss

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=100)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4)
#parser.add_argument('--overlap', action='store_true')
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--clusterfile', type=str, default=None)

sub_args, _ = parser.parse_known_args()

class FindCluster(nn.Module):
    def __init__(self, dim_hids=256, num_inds=32):
        super().__init__()

        self.flow = FlowDistribution(
                MAF(640, dim_hids, 4, dim_context=dim_hids, inv_linear=True),
                Normal(640, use_context=False))
        self.isab1 = StackedISAB(640, dim_hids, num_inds, 4, ln=True, p=0.2)
        self.pma = PMA(dim_hids, dim_hids, 1)
        self.fc1 = nn.Linear(dim_hids, dim_hids)
        nn.init.uniform_(self.fc1.weight, a=-1e-4, b=1e-4)
        nn.init.constant_(self.fc1.bias, 0.0)

        self.mab = MAB(dim_hids, dim_hids, dim_hids)
        self.isab2 = StackedISAB(dim_hids, dim_hids, num_inds, 4)
        self.fc2 = nn.Linear(dim_hids, 1)

    def forward(self, X, mask=None):
        H = self.isab1(X, mask=mask)
        Z = self.pma(H, mask=mask)
        context = self.fc1(Z)
        ll = self.flow.log_prob(X, context).unsqueeze(-1) / 640.0

        H = self.mab(H, Z)
        logits = self.fc2(self.isab2(H, mask=mask))
        return context, ll, logits

class Model(ImageModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.testfile = os.path.join(benchmarks_path,
                'embedded_imagenet_10_100_4.tar' if self.testfile is None else self.testfile)
        self.clusterfile = os.path.join(benchmarks_path,
                'embedded_imagenet_10_300_12.tar' if self.clusterfile is None else self.clusterfile)
        self.net = FindCluster()
        self.metrics = ['ll', 'bcent']

    def get_dataset(self, train=True):
        transform = tvt.Lambda(lambda x: (x - 0.0038)/0.0118)
        return EmbeddedImagenet(ROOT, train=train, transform=transform)

    def get_train_loader(self):
        return get_random_cluster_loader(self.get_dataset(),
                self.B, self.N, self.K, self.num_steps,
                classes=self.get_classes(), mixup=True)

    def loss_fn(self, batch, train=True):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda().float()
        params, ll, logits = self.net(X)
        loss, ll, bcent = compute_filter_loss(ll, logits, labels, lamb=5.0)
        if train:
            return loss
        else:
            return ll, bcent

    def cluster(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            params, ll, logits = self.net(X)
            params = [params]

            labels = torch.zeros_like(logits).squeeze(-1).int()
            mask = (logits > 0.0)
            done = mask.sum((1,2)) == N
            for i in range(1, max_iter):
                params_, ll_, logits = self.net(X, mask=mask)

                ll = torch.cat([ll, ll_], -1)
                params.append(params_)

                ind = logits > 0.0
                labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
                mask[ind] = True

                num_processed = mask.sum((1,2))
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

            fail = done.sum() < B

            # ML estimate of mixing proportion pi
            pi = F.one_hot(labels.long(), len(params)).float()
            pi = pi.sum(1, keepdim=True) / pi.shape[1]
            ll = ll + (pi + 1e-10).log()
            ll = ll.logsumexp(-1).mean()

            if check:
                return params, labels, ll, fail
            else:
                return params, labels, ll

def load(args):
    add_args(args, sub_args)
    return Model(args)
