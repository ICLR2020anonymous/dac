import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
from tqdm import tqdm

from utils.log import Accumulator, get_logger
from utils.paths import benchmarks_path, results_path

from utils.tensor import meshgrid_around, to_numpy
from utils.plots import scatter
import matplotlib.pyplot as plt

from flows.autoregressive import MAF
from flows.distributions import Normal, FlowDistribution

parser = argparse.ArgumentParser()
parser.add_argument('--benchmarkfile', type=str, default='warped_10_3000_12.tar')
parser.add_argument('--filename', type=str, default='test.log')
parser.add_argument('--num_steps', type=int, default=20000)
parser.add_argument('--clip', type=float, default=10.0)
parser.add_argument('--gpu', type=str, default='0')
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train_maf(X):
    flow = FlowDistribution(
            MAF(2, 128, 4),
            Normal(2)).cuda()
    optimizer = optim.Adam(flow.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(r*args.num_steps) for r in [.3, .6]],
            gamma=0.2)

    for i in range(1, args.num_steps+1):
        optimizer.zero_grad()
        loss = -flow.log_prob(X).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), args.clip)

        if i % 1000 == 0:
            print('iter {}, lr {:.3e}, ll {}'.format(
                i, optimizer.param_groups[0]['lr'], -loss.item()))

        optimizer.step()
        scheduler.step()

    return flow.log_prob(X).mean()

benchmark = torch.load(os.path.join(benchmarks_path, args.benchmarkfile))
accm = Accumulator('ll')
save_dir = os.path.join(results_path, 'baselines', 'maf')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
logger = get_logger('baseline_maf', os.path.join(save_dir, args.filename))

for i, dataset in enumerate(benchmark[:10], 1):
    X = dataset['X'].cuda()
    for Xb in X:
        accm.update(train_maf(Xb))
        print()
    print('dataset {} done, avg ll {}'.format(i, accm.get('ll')))
    print()
    logger.info(accm.info())
logger.info(accm.info())
