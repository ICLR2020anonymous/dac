import os
import argparse

import torch
import torch.nn.functional as F

from utils.tensor import to_numpy

from models.base import ModelTemplate

from data.paired_dataset import (PairedDataset,
        sample_pairs, get_random_pair_loader,
        get_saved_pair_loader)

class ImagePairModelTemplate(ModelTemplate):
    def __init__(self, args):
        super().__init__(args)
        self.metrics = ['bcent']

    def get_dataset(self, train=True):
        raise NotImplementedError

    def get_classes(self, train=True):
        return None

    def gen_benchmarks(self, force=False):
        dataset = PairedDataset(
                self.get_dataset(False),
                classes=self.get_classes(False))

        def sample_batch():
            return sample_pairs(dataset.idx_map, self.batch_size)

        if not os.path.isfile(self.testfile) or force:
            print('generating benchmark {}...'.format(self.testfile))
            bench = [sample_batch() for _ in range(100)]
            torch.save(bench, self.testfile)

    def loss_fn(self, batch, train=True):
        logits = self.net(batch['X1'].cuda(), batch['X2'].cuda())
        return F.binary_cross_entropy_with_logits(logits, batch['labels'].float().cuda())

    def get_train_loader(self):
        return get_random_pair_loader(self.get_dataset(),
                self.batch_size, self.num_steps,
                classes=self.get_classes())

    def get_test_loader(self, filename=None):
        return get_saved_pair_loader(self.get_dataset(False),
                (self.testfile if filename is None else filename),
                classes=self.get_classes(False))
