import numpy as np
import pickle
import os

import torch
from torchvision.datasets.vision import VisionDataset
ROOT = '/mnt/banner/shared/LEO_embeddings'

class EmbeddedImagenet(VisionDataset):

    def check_if_raw_exists(self):
        if self.check_if_preprocessed():
            return

        if not os.path.isdir(os.path.join(self.root, 'embeddings')):
            if not os.path.isfile(os.path.join(self, root, 'embeddings.zip')):
                raise NotImplemented

    def check_if_preprocessed(self):
        return os.path.isfile(os.path.join(self.root, 'train.pt')) and \
                os.path.isfile(os.path.join(self.root, 'test.pt'))

    def preprocess(self):
        train_list = ['embeddings/tieredImageNet/center/train_embeddings.pkl',
                'embeddings/tieredImageNet/center/val_embeddings.pkl',
                'embeddings/miniImageNet/center/train_embeddings.pkl',
                'embeddings/miniImageNet/center/val_embeddings.pkl']

        data = []
        targets = []
        labels_dict = {}
        l = 0
        for filename in train_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')

            data_ = torch.Tensor(entry['embeddings'])
            for key in entry['keys']:
                label_code = key.split('-')[1]
                if labels_dict.get(label_code, None) == None:
                    labels_dict[label_code] = l
                    l += 1

            targets_ = torch.LongTensor(len(data_))
            for i, key in enumerate(entry['keys']):
                targets_[i] = labels_dict[key.split('-')[1]]

            data.append(data_)
            targets.append(targets_)

        data = torch.cat(data, 0)
        targets = torch.cat(targets, 0)
        torch.save((data, targets), os.path.join(self.root, 'train.pt'))

        test_list = ['embeddings/tieredImageNet/center/test_embeddings.pkl',
                'embeddings/miniImageNet/center/test_embeddings.pkl']
        data = []
        targets = []
        labels_dict = {}
        l = 0
        for filename in test_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')

            data_ = torch.Tensor(entry['embeddings'])
            for key in entry['keys']:
                label_code = key.split('-')[1]
                if labels_dict.get(label_code, None) == None:
                    labels_dict[label_code] = l
                    l += 1

            targets_ = torch.LongTensor(len(data_))
            for i, key in enumerate(entry['keys']):
                targets_[i] = labels_dict[key.split('-')[1]]

            data.append(data_)
            targets.append(targets_)

        data = torch.cat(data, 0)
        targets = torch.cat(targets, 0)
        torch.save((data, targets), os.path.join(self.root, 'test.pt'))

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, transform=transform)

        self.check_if_raw_exists()
        if not self.check_if_preprocessed():
            self.preprocess()

        self.data, self.targets = torch.load(
                os.path.join(self.root,
                    'train.pt' if train else 'test.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

if __name__ == '__main__':

    dataset = EmbeddedImagenet(ROOT, train=True)

    print(torch.unique(dataset.targets))
