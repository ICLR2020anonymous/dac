import numpy as np
import pickle
import os
from PIL import Image

import torch
from torchvision.datasets.vision import VisionDataset
ROOT = '/mnt/banner/shared/ImageNet32'
MEAN = [0.4810, 0.4574, 0.4078]
STD = [0.2605, 0.2533, 0.2684]

class Imagenet32(VisionDataset):
    train_list = ['train_data_batch_{}'.format(i+1) for i in range(7)]
    val_list = ['val_data'] + ['train_data_batch_{}'.format(i+1) for i in range(7, 10)]
    def __init__(self, root, train=True, transform=None,
            classes=None):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, 'processed', filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, 32, 32))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose(0, 2, 3, 1)
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as tvt

    data = Imagenet32(ROOT, train=True, classes=range(800), transform=tvt.ToTensor())
    loader = DataLoader(data)
    labels = []
    for x, y in loader:
        labels.append(y.numpy())

    print(np.sort(np.unique(np.concatenate(labels))))
