import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from data.cluster import sample_labels

def sample_idxs_and_labels(idx_map, B, N, K, rand_N=True, rand_K=True, mixup=False):
    N = np.random.randint(int(0.3*N), N) if rand_N else N
    labels = sample_labels(B, N, K, rand_K=rand_K, device='cpu', alpha=5.0)

    def sample_idxs(labels):
        idxs = torch.zeros(B, N, dtype=torch.long)
        classes = [0]*B
        classes_pool = list(idx_map.keys())
        for b in range(B):
            classes[b] = np.random.permutation(classes_pool)[:K]
            for i, l in enumerate(classes[b]):
                if (labels[b] == i).int().sum() > 0:
                    members = (labels[b] == i).nonzero().view(-1)
                    idx_pool = idx_map[l]
                    idx_pool = idx_pool[torch.randperm(len(idx_pool))]
                    n_repeat = len(members) // len(idx_pool) + 1
                    idxs[b, members] = torch.cat([idx_pool]*n_repeat)[:len(members)]
        return idxs, classes

    batch = {}
    if mixup:
        batch['idxs1'], batch['classes1'] = sample_idxs(labels)
        batch['idxs2'], batch['classes2'] = sample_idxs(labels)
    else:
        batch['idxs'], batch['classes'] = sample_idxs(labels)
    batch['labels'] = F.one_hot(labels, K)
    return batch

class ClusteredDataset(object):
    def __init__(self, dataset, classes=None):
        self.dataset = dataset
        if not type(self.dataset.targets) == torch.Tensor:
            self.dataset.targets = torch.Tensor(self.dataset.targets)

        if classes is None:
            self.classes = torch.unique(self.dataset.targets).numpy()
        else:
            self.classes = np.array(classes)
        self.num_classes = len(self.classes)

        self.idx_map = {l:(dataset.targets==l).nonzero().squeeze() \
                for l in self.classes}

    def __getitem__(self, batch):
        if batch.get('idxs1', None) is None:
            idxs = batch.pop('idxs')
            X = (self.dataset[i] for i in idxs.flatten())
            X = torch.stack([x for x, _, in X])
            _, *shape = X.shape
            batch['X'] = X.reshape(*idxs.shape, *shape)
        else:
            idxs1, idxs2 = batch.pop('idxs1'), batch.pop('idxs2')
            lam = torch.Tensor(np.random.beta(0.2, 0.2, len(idxs1)))
            X1 = torch.stack([self.dataset[i][0] for i in idxs1.flatten()])
            X2 = torch.stack([self.dataset[i][0] for i in idxs2.flatten()])

            _, *shape = X1.shape
            X1 = X1.reshape(*idxs1.shape, *shape)
            X2 = X2.reshape(*idxs2.shape, *shape)
            lam = lam.view((-1,) + (1,)*(X1.dim()-1))
            batch['X'] = lam*X1 + (1-lam)*X2
        return batch

class RandomClusterSampler(object):
    def __init__(self, sample_fn, num_steps):
        self.num_steps = num_steps
        self.sample_fn = sample_fn

    def __iter__(self):
        for _ in range(self.num_steps):
            yield [self.sample_fn()]

    def __len__(self):
        return self.num_steps

class SavedClusterSampler(object):
    def __init__(self, filename):
        self.batches = [[batch] for batch in torch.load(filename)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_random_cluster_loader(dataset, B, N, K, num_steps,
        classes=None, **kwargs):
    dataset = ClusteredDataset(dataset, classes=classes)
    sample_fn = lambda : sample_idxs_and_labels(dataset.idx_map,
            B, N, K, **kwargs)
    sampler = RandomClusterSampler(sample_fn, num_steps)
    return DataLoader(dataset, num_workers=4, batch_sampler=sampler,
            collate_fn=lambda x: x[0])

def get_saved_cluster_loader(dataset, filename, classes=None):
    dataset = ClusteredDataset(dataset, classes=classes)
    sampler = SavedClusterSampler(filename)
    return DataLoader(dataset, num_workers=4, batch_sampler=sampler,
            collate_fn=lambda x: x[0])

if __name__ == '__main__':

    from data.mnist import FastMNIST
    from data.emnist import FastEMNIST
    from data.kkanji import KKanji
    import os
    from utils.paths import datasets_path
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = FastEMNIST(os.path.join(datasets_path, 'emnist'), train=True,
            split='balanced')
    #dataset = Kkanji(os.path.join(datasets_path, 'kkanji'), train=True)
    loader = get_random_cluster_loader(dataset, 10, 100, 4, 1,
            classes=[18,11,4,6,8], rand_N=False, rand_K=False, mixup=True)

    batch = next(iter(loader))

    X = batch['X']
    print(X.shape)
    print(batch['classes1'])
    print(batch['classes2'])
    labels = batch['labels'].argmax(-1)
    print(labels)

    plt.imshow(make_grid(X[0][labels[0]==3]).numpy().transpose(1,2,0))
    plt.show()
