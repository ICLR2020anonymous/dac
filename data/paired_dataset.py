import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def sample_pairs(idx_map, batch_size, ratio=0.5):

    num_same = int(batch_size * ratio)
    num_diff = batch_size - num_same
    classes = list(idx_map.keys())

    idxs = torch.zeros(batch_size, 2).long()
    labels = torch.zeros(batch_size).long()
    for i in range(num_same):
        c = np.random.permutation(classes)[0]
        idxs[i] = idx_map[c][np.random.randint(0, len(idx_map[c]), [2])]
        labels[i] = 1

    for i in range(num_same, batch_size):
        c1, c2 = np.random.permutation(classes)[:2]
        idxs[i,0] = idx_map[c1][np.random.randint(len(idx_map[c1]))]
        idxs[i,1] = idx_map[c2][np.random.randint(len(idx_map[c2]))]
        labels[i] = 0

    order = torch.randperm(batch_size)
    return {'idxs':idxs[order], 'labels':labels[order]}

class PairedDataset(object):
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
        idxs = batch.pop('idxs')
        idxs1, idxs2 = idxs[...,0], idxs[...,1]
        X1 = (self.dataset[i] for i in idxs1.flatten())
        X1 = torch.stack([x for x, _, in X1])
        X2 = (self.dataset[i] for i in idxs2.flatten())
        X2 = torch.stack([x for x, _, in X2])
        _, *shape = X1.shape
        batch['X1'] = X1.reshape(*idxs1.shape, *shape)
        batch['X2'] = X2.reshape(*idxs2.shape, *shape)
        return batch

class RandomPairSampler(object):
    def __init__(self, sample_fn, num_steps):
        self.num_steps = num_steps
        self.sample_fn = sample_fn

    def __iter__(self):
        for _ in range(self.num_steps):
            yield [self.sample_fn()]

    def __len__(self):
        return self.num_steps

class SavedPairSampler(object):
    def __init__(self, filename):
        self.batches = [[batch] for batch in torch.load(filename)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_random_pair_loader(dataset, batch_size, num_steps,
        classes=None, **kwargs):
    dataset = PairedDataset(dataset, classes=classes)
    sample_fn = lambda : sample_pairs(dataset.idx_map, batch_size, **kwargs)
    sampler = RandomPairSampler(sample_fn, num_steps)
    return DataLoader(dataset, num_workers=4, batch_sampler=sampler,
            collate_fn=lambda x: x[0])

def get_saved_pair_loader(dataset, filename, classes=None):
    dataset = PairedDataset(dataset, classes=classes)
    sampler = SavedPairSampler(filename)
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

    #dataset = FastEMNIST(os.path.join(datasets_path, 'emnist'), train=True,
    #        split='balanced')
    dataset = KKanji(os.path.join(datasets_path, 'kkanji'), train=True)
    loader = get_random_pair_loader(dataset, 50, 1)
    batch = next(iter(loader))

    X1 = batch['X1']
    X2 = batch['X2']

    print(batch['labels'])

    plt.figure()
    plt.imshow(make_grid(X1).numpy().transpose(1,2,0))
    plt.figure()
    plt.imshow(make_grid(X2).numpy().transpose(1,2,0))
    plt.show()
