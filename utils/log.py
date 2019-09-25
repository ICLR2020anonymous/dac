import torch
import time
import logging
import numpy as np

def get_logger(header, filename):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(header)
    logger.addHandler(logging.FileHandler(filename, mode='w'))
    return logger

class Accumulator():
    def __init__(self, *args):
        self.args = args
        self.argnum = {}
        for i, arg in enumerate(args):
            self.argnum[arg] = i
        self.sums = [0]*len(args)
        self.cnt = 0
        self.clock = time.time()

    def update(self, val):
        try:
            iter(val)
        except:
            val = [val]
        else:
            val = val
        val = [v for v in val if v is not None]
        for i, v in enumerate(val):
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.sums[i] += v
        self.cnt += 1

    def reset(self):
        self.sums = [0]*len(self.args)
        self.cnt = 0
        self.clock = time.time()

    def get(self, arg):
        i = self.argnum.get(arg)
        if i is not None:
            return self.sums[i]/self.cnt
        else:
            return None

    def info(self, header=None, epoch=None, it=None, show_et=True):
        et = time.time() - self.clock
        line = '' if header is None else header + ': '
        if epoch is not None:
            line += 'epoch {:d}, '.format(epoch)
        if it is not None:
            line += 'iter {:d}, '.format(it)
        for arg in self.args:
            val = self.sums[self.argnum[arg]]/self.cnt
            if type(val) == np.float64 or type(val) == float:
                line += '{} {:.4f}, '.format(arg, val)
            else:
                line += '{} {}, '.format(arg, val)
        if show_et:
            line += '({:.3f} secs)'.format(et)
        return line
