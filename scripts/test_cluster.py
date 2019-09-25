import torch
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import numpy as np

from utils.log import get_logger, Accumulator
from utils.misc import load_module
from utils.paths import results_path
from utils.tensor import to_numpy

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str, default='models/mog.py')
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--max_iter', type=int, default=50)
parser.add_argument('--filename', type=str, default='test_cluster.log')
parser.add_argument('--gpu', type=str, default='0')
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
net.eval()
test_loader = model.get_test_loader(filename=model.clusterfile)
accm = Accumulator('model ll', 'oracle ll', 'ARI', 'NMI', 'k-MAE')
num_failure = 0
logger = get_logger('{}_{}'.format(module_name, args.run_name),
        os.path.join(save_dir, args.filename))
for batch in tqdm(test_loader):
    params, labels, ll, fail = model.cluster(batch['X'].cuda(),
            max_iter=args.max_iter, verbose=False, check=True)
    true_labels = to_numpy(batch['labels'].argmax(-1))
    ari = 0
    nmi = 0
    mae = 0
    for b in range(len(labels)):
        labels_b = to_numpy(labels[b])
        ari += ARI(true_labels[b], labels_b)
        nmi += NMI(true_labels[b], labels_b, average_method='arithmetic')
        mae += abs(len(np.unique(true_labels[b])) - len(np.unique(labels_b)))
    ari /= len(labels)
    nmi /= len(labels)
    mae /= len(labels)

    oracle_ll = 0.0 if batch.get('ll') is None else batch['ll']
    accm.update([ll.item(), oracle_ll, ari, nmi, mae])
    num_failure += int(fail)

logger.info(accm.info())
logger.info('number of failure cases {}'.format(num_failure))
