import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as WN

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

"""
Largely adpated from
https://github.com/bayesiains/nsf
https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
"""

def build_mask(dim_inputs, dim_outputs, dim_flows, mask_type=None):
    """ mask_type: input | None | output
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(dim_inputs) % dim_flows
    else:
        in_degrees = torch.arange(dim_inputs) % (dim_flows - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(dim_outputs) % dim_flows - 1
    else:
        out_degrees = torch.arange(dim_outputs) % (dim_flows - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, mask):
        super().__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)
        self.register_buffer('mask', mask)

    def forward(self, x):
        output = F.linear(x, self.linear.weight * self.mask, self.linear.bias)
        return output

class ConvResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                    stride=stride, bias=False)

        self.block = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 3,
                    padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3,
                    padding=1, bias=False))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

#class ConvResUnit(nn.Module):
#    def __init__(self, in_channels, out_channels,
#            hid_channels=None, stride=1,
#            weight_norm=False,
#            batch_norm=False,
#            zero_init=False):
#        super().__init__()
#
#        fn = WN if weight_norm else lambda x: x
#        if in_channels != out_channels or stride != 1:
#            self.shortcut = fn(nn.Conv2d(in_channels, out_channels, 3,
#                    padding=1, stride=stride, bias=False))
#        else:
#            self.shortcut = nn.Identity()
#
#        hid_channels = hid_channels or out_channels
#        self.block = nn.Sequential(
#                nn.ELU(),
#                fn(nn.Conv2d(in_channels, hid_channels, 3,
#                    padding=1, stride=stride)),
#                nn.ELU(),
#                fn(nn.Conv2d(hid_channels, out_channels, 3, padding=1)))
#
#        if zero_init:
#            if weight_norm:
#                self.block[-1].weight_g.data.zero_()
#            else:
#                self.block[-1].weight.data.zero_()
#            self.block[-1].bias.data.zero_()
#
#    def forward(self, x, context=None):
#        return self.shortcut(x) + self.block(x)
#
#class DeconvResUnit(nn.Module):
#    def __init__(self, in_channels, out_channels,
#            hid_channels=None, stride=1,
#            weight_norm=False, zero_init=False):
#        super().__init__()
#
#        output_padding = 1 if stride > 1 else 0
#
#        fn = WN if weight_norm else lambda x: x
#        if in_channels != out_channels or stride != 1:
#            self.shortcut = fn(nn.ConvTranspose2d(in_channels, out_channels, 3,
#                padding=1, output_padding=output_padding,
#                stride=stride, bias=False))
#        else:
#            self.shortcut = nn.Identity()
#
#        hid_channels = hid_channels or out_channels
#        fn = WN if weight_norm else lambda x: x
#        self.block = nn.Sequential(
#                nn.ELU(),
#                fn(nn.ConvTranspose2d(in_channels, hid_channels, 3,
#                    padding=1, output_padding=output_padding, stride=stride)),
#                nn.ELU(),
#                fn(nn.ConvTranspose2d(hid_channels, out_channels, 3, padding=1)))
#
#        if zero_init:
#            if weight_norm:
#                self.block[-1].weight_g.data.zero_()
#            else:
#                self.block[-1].weight.data.zero_()
#            self.block[-1].bias.data.zero_()
#
#    def forward(self, x, context=None):
#        return self.shortcut(x) + self.block(x)

class FixupResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                padding=1, stride=stride, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                    stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.elu(x)
        out = self.conv1(out + self.bias1a)
        out = out + self.bias1b

        out = F.elu(out)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        return self.shortcut(x) + out
