import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.tensor import chunk_two, sum_except_batch as sumeb
from flows.transform import Transform, Composite
from flows.permutations import Flip

def batched_get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(0) >= in_degrees.unsqueeze(-1)).float()

class BatchedMaskedLinear(nn.Module):
    def __init__(self, batch_size, in_features, out_features, mask):
        super().__init__()
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.Tensor(batch_size, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(batch_size, 1, out_features))
        self.register_buffer('mask', mask[None])
        self.reset_parameters()

    def reset_parameters(self):
        for b in range(self.batch_size):
            nn.init.kaiming_uniform_(self.weight[b], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[b])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[b], -bound, bound)

    def forward(self, X):
        return torch.matmul(X, self.weight*self.mask) + self.bias

class BatchedMADE(Transform):
    def __init__(self, batch_size, dim_inputs, dim_hids):
        super().__init__()

        self.dim = dim_inputs

        input_mask = batched_get_mask(dim_inputs, dim_hids,
                dim_inputs, mask_type='input')
        hidden_mask = batched_get_mask(dim_hids, dim_hids, dim_inputs)
        output_mask = batched_get_mask(dim_hids, 2*dim_inputs,
                dim_inputs, mask_type='output')

        self.mlp = nn.Sequential(
                BatchedMaskedLinear(batch_size, dim_inputs, dim_hids, input_mask),
                nn.ELU(),
                BatchedMaskedLinear(batch_size, dim_hids, dim_hids, hidden_mask),
                nn.ELU(),
                BatchedMaskedLinear(batch_size, dim_hids, 2*dim_inputs, output_mask))

    def get_params(self, X):
        H = self.mlp(X)
        shift, scale = chunk_two(H)
        scale = F.softplus(scale) + 1e-5
        return shift, scale

    def forward(self, X, context=None):
        shift, scale = self.get_params(X)
        return (X - shift)/scale, -sumeb(scale.log())

    def inverse(self, Z, context=None):
        X = torch.zeros_like(Z)
        for i in range(self.dim):
            shift, scale = self.get_params(X)
            X[...,i] = Z[...,i] * scale[...,i] + shift[...,i]
        return X, sumeb(scale.log())

class BatchedMAF(Composite):
    def __init__(self, batch_size, dim_inputs, dim_hids, num_blocks):
        transforms = []
        for _ in range(num_blocks):
            transforms.append(BatchedMADE(batch_size, dim_inputs, dim_hids))
            transforms.append(Flip())
        super().__init__(transforms)
