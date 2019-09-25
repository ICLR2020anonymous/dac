import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.tensor import to_numpy

def compute_filter_loss(ll, logits, labels, lamb=1.0):
    B, K = labels.shape[0], labels.shape[-1]
    bcent = F.binary_cross_entropy_with_logits(
            logits.repeat(1, 1, K),
            labels, reduction='none').mean(1)
    ll = (ll * labels).sum(1) / (labels.sum(1) + 1e-8)
    loss = lamb * bcent - ll
    loss[ll==0] = float('inf')
    loss, idx = loss.min(1)
    bidx = loss != float('inf')

    loss = loss[bidx].mean()
    ll = ll[bidx, idx[bidx]].mean()
    bcent = bcent[bidx, idx[bidx]].mean()
    return loss, ll, bcent

class ModelTemplate(object):
    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.net = None
        self.metrics = ['ll', 'bcent']

    def load_from_ckpt(self):
        pass

    def sample(self, B, N, K):
        raise NotImplementedError

    def get_train_loader(self):
        for _ in range(self.num_steps):
            yield self.sample(self.B, self.N, self.K)

    def gen_benchmarks(self, force=False):
        pass

    def get_test_loader(self, filename=None):
        filename = self.testfile if filename is None else filename
        return torch.load(filename)

    def build_optimizer(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.num_steps)
        return optimizer, scheduler

    # compute training loss
    def loss_fn(self, batch, train=True, lamb=1.0):
        X = batch['X'].cuda()
        labels = batch['labels'].cuda().float()
        params, ll, logits = self.net(X)
        loss, ll, bcent = compute_filter_loss(ll, logits, labels, lamb=lamb)
        if train:
            return loss
        else:
            return ll, bcent

    def cluster(self, X, max_iter=50, verbose=True, check=False):
        B, N = X.shape[0], X.shape[1]
        self.net.eval()

        with torch.no_grad():
            params, ll, logits = self.net(X)
            params = [params]

            labels = torch.zeros_like(logits).squeeze(-1).int()
            mask = (logits > 0.0)
            done = mask.sum((1,2)) == N
            for i in range(1, max_iter):
                params_, ll_, logits = self.net(X, mask=mask)

                ll = torch.cat([ll, ll_], -1)
                params.append(params_)

                ind = logits > 0.0
                labels[(ind*mask.bitwise_not()).squeeze(-1)] = i
                mask[ind] = True

                num_processed = mask.sum((1,2))
                done = num_processed == N
                if verbose:
                    print(to_numpy(num_processed))
                if done.sum() == B:
                    break

            fail = done.sum() < B

            # ML estimate of mixing proportion pi
            pi = F.one_hot(labels.long(), len(params)).float()
            pi = pi.sum(1, keepdim=True) / pi.shape[1]
            ll = ll + (pi + 1e-10).log()
            ll = ll.logsumexp(-1).mean()

            if check:
                return params, labels, ll, fail
            else:
                return params, labels, ll

    def plot_clustering(self, X, params, labels):
        raise NotImplementedError

    def plot_step(self, X):
        raise NotImplementedError
