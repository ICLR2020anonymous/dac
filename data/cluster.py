import torch
from torch.distributions import Dirichlet, Categorical
import math

def sample_labels(B, N, K, alpha=1.0, rand_K=True, device='cpu'):
    pi = Dirichlet(alpha*torch.ones(K)).sample([B]).to(device)
    if rand_K:
        to_use = (torch.rand(B, K) < 0.5).float().to(device)
        to_use[...,0] = 1
        pi = pi * to_use
        pi = pi/pi.sum(1, keepdim=True)
    labels = Categorical(probs=pi).sample([N]).to(device)
    labels = labels.transpose(0,1).contiguous()
    return labels

def sample_masks(labels):
    B, K = labels.shape[0], labels.shape[-1]
    to_mask = (torch.rand(B, 1, K) < 0.3).float().to(labels.device)
    to_mask[...,0] = 0
    mask = (labels.float() * to_mask).sum(-1, keepdim=True).byte()
    return mask

def sample_anchors(B, N, mask=None, device='cpu'):
    if mask is None:
        return torch.randint(0, N, [B]).to(device)
    else:
        mask = mask.view(B, N)
        anchor_idxs = torch.zeros(B, dtype=torch.int64).to(device)
        for b in range(B):
            if mask[b].sum() < N:
                idx_pool = mask[b].bitwise_not().nonzero().view(-1)
                anchor_idxs[b] = idx_pool[torch.randint(len(idx_pool), [1])]
        return anchor_idxs
