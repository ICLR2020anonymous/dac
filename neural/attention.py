import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_X, dim_Y, dim,
            num_heads=4, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_X, dim)
        self.fc_k = nn.Linear(dim_Y, dim)
        self.fc_v = nn.Linear(dim_Y, dim)
        self.fc_o = nn.Linear(dim, dim)

        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()

    def forward(self, X, Y, mask=None):
        Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)

        A_logits = (Q_ @ K_.transpose(1,2)) / math.sqrt(Q.shape[-1])
        if mask is not None:
            mask = mask.squeeze(-1).unsqueeze(1)
            mask = mask.repeat(self.num_heads, Q.shape[1], 1)
            A_logits.masked_fill_(mask, -100.0)
        A = torch.softmax(A_logits, -1)

        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.ln1(Q + self.dropout1(attn))
        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        return O

class SAB(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.mab = MAB(dim_X, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class StackedSAB(nn.Module):
    def __init__(self, dim_X, dim, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [SAB(dim_X, dim, **kwargs)] + \
                [SAB(dim, dim, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for sab in self.blocks:
            X = sab(X, mask=mask)
        return X

class PMA(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(self.I.repeat(X.shape[0], 1, 1), X, mask=mask)

class ISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.pma = PMA(dim_X, dim, num_inds, **kwargs)
        self.mab = MAB(dim_X, dim, dim, **kwargs)

    def forward(self, X, mask=None):
        return self.mab(X, self.pma(X, mask=mask))

class StackedISAB(nn.Module):
    def __init__(self, dim_X, dim, num_inds, num_blocks, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
                [ISAB(dim_X, dim, num_inds, **kwargs)] + \
                [ISAB(dim, dim, num_inds, **kwargs)]*(num_blocks-1))

    def forward(self, X, mask=None):
        for isab in self.blocks:
            X = isab(X, mask=mask)
        return X

class aPMA(nn.Module):
    def __init__(self, dim_X, dim, **kwargs):
        super().__init__()
        self.I0 = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.I0)
        self.pma = PMA(dim, dim, 1, **kwargs)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, num_iters):
        I = self.I0
        for i in range(1, num_iters):
            I = torch.cat([I, self.pma(I)], 1)
        return self.mab(I.repeat(X.shape[0], 1, 1), X)

# copied & adapted from
# https://github.com/leaderj1001/Stand-Alone-Self-Attention
# https://github.com/MerHS/SASA-pytorch
# Implementation of standalone self-attention layer
# Ramachandran et al. 2019. Stand-Alone Self-Attention in Vision Models
class SelfAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0

        self.rel_size = (out_channels//groups) // 2
        self.rel_x = nn.Parameter(torch.Tensor(self.rel_size, kernel_size))
        self.rel_y = nn.Parameter(torch.Tensor(out_channels//groups - self.rel_size, kernel_size))

        self.query_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1)) \
                if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.rel_x, 0, 1)
        nn.init.normal_(self.rel_y, 0, 1)

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, [self.padding]*4)

        ksize, stride = self.kernel_size, self.stride
        pH, pW = H + self.padding*2, W + self.padding*2
        fH, fW = (pH - ksize)//stride + 1, (pW - ksize)//stride + 1

        # B * fC * pH, * pW
        q, k, v = self.query_conv(x), self.key_conv(x), self.value_conv(x)
        win_q = q[:,:,
                (ksize-1)//2:pH-(ksize//2):stride,
                (ksize-1)//2:pW-(ksize//2):stride]
        win_q_b = win_q.view(B, self.groups, -1, fH, fW)
        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=-3)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.rel_x))
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.rel_y))

        win_k = k.unfold(2, ksize, stride).unfold(3, ksize, stride)

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(1)
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4)
        vx = torch.softmax(vx.view(B, fH, fW, -1), 3).view(B, 1, fH, fW, ksize, ksize)

        win_v = v.unfold(2, ksize, stride).unfold(3, ksize, stride)
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, ))

        if self.bias is not None:
            fin_v += self.bias
        return fin_v

class SelfAttentionResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = SelfAttentionConv2d(in_channels, out_channels, 7,
                padding=3, groups=groups, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SelfAttentionConv2d(out_channels, out_channels, 7,
                padding=3, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #self.pool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return self.shortcut(x) + out
