import os
import argparse
import torch
import torch.nn as nn

from utils.paths import datasets_path
from utils.misc import add_args

from neural.modules import ConvResUnit, DeconvResUnit

from flows.transform import Composite
from flows.coupling import AffineCoupling
from flows.image import Dequantize
from flows.permutations import Invertible1x1Conv
#from flows.glow import Glow
from flows.distributions import Normal, FlowDistribution

from models.base import ModelTemplate

import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from data.mini_imagenet import MiniImagenet

# for training
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_steps', type=int, default=80000)

sub_args, _ = parser.parse_known_args()

class SimpleFlowDist(FlowDistribution):
    def __init__(self, shape, num_blocks, use_context=False):
        net_fn = lambda C_in, C_out, d_ctx: ConvResUnit(C_in, C_out, zero_init=True)
        transforms = []
        for _ in range(num_blocks):
            transforms.append(AffineCoupling(shape[0], net_fn))
            transforms.append(Invertible1x1Conv(shape[0]))
        super().__init__(Composite(transforms),
                Normal(shape, use_context=use_context))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                # to 32 * 32 * 32
                ConvResUnit(3, 32, stride=2, weight_norm=True),
                ConvResUnit(32, 32, weight_norm=True),

                # to 64 * 16 * 16
                ConvResUnit(32, 64, stride=2, weight_norm=True),
                ConvResUnit(64, 64, weight_norm=True),

                # to 64 * 8 * 8
                ConvResUnit(64, 64, stride=2, weight_norm=True),
                ConvResUnit(64, 64, weight_norm=True),

                # to (64 + 64) * 4 * 4
                ConvResUnit(64, 64, stride=2, weight_norm=True),
                ConvResUnit(64, 128, weight_norm=True)
                )

        self.posterior = Normal((64, 4, 4), use_context=True)
        #self.posterior = SimpleFlowDist((64, 4, 4), 4, use_context=True)
        self.prior = SimpleFlowDist((64, 4, 4), 4, use_context=False)

        self.decoder = nn.Sequential(
                # to 64 * 8 * 8
                DeconvResUnit(64, 64, weight_norm=True),
                DeconvResUnit(64, 64, stride=2, weight_norm=True),

                # to 64 * 16 * 16
                DeconvResUnit(64, 64, weight_norm=True),
                DeconvResUnit(64, 64, stride=2, weight_norm=True),

                # to 32 * 32 * 32
                DeconvResUnit(64, 64, weight_norm=True),
                DeconvResUnit(64, 32, stride=2, weight_norm=True),

                # to (3 + 3) * 64 * 64
                DeconvResUnit(32, 32, weight_norm=True),
                DeconvResUnit(32, 6, stride=2)
                )

        self.likel = FlowDistribution(Dequantize(),
                Normal((3, 64, 64), use_context=True))

    def forward(self, x):
        _, C, H, W = x.shape
        h_enc = self.encoder(x)
        z, log_q = self.posterior.sample(context=h_enc)
        log_p = self.prior.log_prob(z)
        kld = (log_q - log_p).mean()/(C*H*W)
        h_dec = self.decoder(z)
        ll = self.likel.log_prob(x, context=h_dec).mean()/(C*H*W)
        return ll, kld

    def generate(self, num_samples, device='cpu'):
        z, _ = self.prior.sample(num_samples, device=device)
        h_dec = self.decoder(z)
        x, _ = self.likel.sample(context=h_dec)
        return x

    def reconstruct(self, x):
        h_enc = self.encoder(x)
        z, _ = self.posterior.mean(context=h_enc)
        h_dec = self.decoder(z)
        x, _ = self.likel.mean(context=h_dec)
        return x

class Model(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.net = VAE()
        self.metrics = ['ll/p', 'kld/p', 'elbo/p']

    def get_train_loader(self):
        dataset = MiniImagenet(os.path.join(datasets_path, 'mini-imagenet'),
                split='train', transform=tvt.ToTensor())
        loader = DataLoader(dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=4)
        num_epochs = self.num_steps // len(loader)
        for _ in range(num_epochs):
            for x, y in loader:
                yield x, y

    def get_test_loader(self):
        dataset = MiniImagenet(os.path.join(datasets_path, 'mini-imagenet'),
                split='test_overlap', transform=tvt.ToTensor())
        return DataLoader(dataset, batch_size=self.test_batch_size,
                num_workers=4)

    def loss_fn(self, dataset, train=True):
        x, _ = dataset
        x = x.cuda()
        ll, kld = self.net(x)
        elbo = ll - kld
        if train:
            return -elbo
        else:
            return ll, kld, elbo

    def reconstruct(self, x):
        self.net.eval()
        with torch.no_grad():
            return self.net.reconstruct(x)

    def generate(self, num_samples, device='cpu'):
        self.net.eval()
        with torch.no_grad():
            return self.net.generate(num_samples, device=device)

def load(args):
    add_args(args, sub_args)
    return Model(args)

if __name__ == '__main__':
    x = torch.rand(32, 3, 64, 64)
    model = Model(sub_args)
    model.net(x)
