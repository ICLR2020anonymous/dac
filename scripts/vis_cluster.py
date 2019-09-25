import os
import argparse

import torch

from utils.paths import results_path, benchmarks_path
from utils.misc import load_module
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--rand_N', action='store_true')
parser.add_argument('--rand_K', action='store_true')
parser.add_argument('--mode', type=str, default='cluster',
        choices=['cluster', 'step'])
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--save', action='store_true')
parser.add_argument('--filename', type=str, default=None)
args, _ = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

module, module_name = load_module(args.modelfile)
model = module.load(args)
print(str(args))

if not hasattr(model, 'cluster'):
    raise ValueError('Model is not for clustering')

save_dir = os.path.join(results_path, module_name, args.run_name)
net = model.net.cuda()

net.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar')))

batch = model.sample(args.vB, args.vN, args.vK,
        rand_N=args.rand_N, rand_K=args.rand_K)
X = batch['X']

if args.mode == 'cluster':
    params, labels, ll = model.cluster(X.cuda(), max_iter=args.max_iter)
    print('plotting...')
    model.plot_clustering(X, params, labels)
    print('log likelihood: {}'.format(ll.item()))
elif args.mode == 'step':
    model.plot_step(X.cuda())

if args.save:
    if not os.path.isdir('./figures'):
        os.makedirs('./figures')
    filename = os.path.join('./figures', '{}_{}.pdf'.format(module_name, args.mode)) \
            if args.filename is None else args.filename
    plt.savefig(filename, bbox_inches='tight')
plt.show()
