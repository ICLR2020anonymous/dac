import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.modules import ConvResUnit
from flows.transform import Transform, Composite
from flows.coupling import AffineCoupling
from flows.normalizations import ActNorm
from flows.permutations import Invertible1x1Conv
from flows.distributions import Normal
from flows.image import Squeeze, MultiscaleComposite, Dequantize

class GlowTransNet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, dim_context=0):
        super().__init__()
        self.net = ConvResUnit(in_channels, out_channels,
                hid_channels=hid_channels, zero_init=True)
        self.ctx_linear = nn.Linear(dim_context, out_channels, bias=False) \
                if dim_context > 0 else None

    def forward(self, x, context=None):
        x = self.net(x)
        if context is not None and self.ctx_linear is not None:
            x = x + self.ctx_linear(x)[...,None,None]
        return x

class GlowBlock(Composite):
    def __init__(self, shape, hid_channels, num_steps, dim_context=0):
        net_fn = lambda C_in, C_out, d_ctx: \
                GlowTransNet(C_in, C_out, hid_channels, d_ctx)
        transforms = [Squeeze()]
        C, H, W = transforms[0].output_shape(shape)
        for _ in range(num_steps):
            transforms.append(ActNorm(C, dim_context))
            transforms.append(Invertible1x1Conv(C))
            transforms.append(AffineCoupling(C, net_fn))
        self.output_shape = (C, H, W)
        super().__init__(transforms)

class Glow(Composite):
    def __init__(self, shape, hid_channels, num_blocks, num_steps,
            dim_context=0, num_bits=8, multiscale=False):
        C, H, W = shape
        transforms = [Dequantize(num_bits=num_bits)]
        if multiscale:
            msc = MultiscaleComposite()
            for i in range(num_blocks):
                block = GlowBlock(shape, hid_channels, num_steps, dim_context)
                shape = msc.append(block, block.output_shape, last=(i == num_blocks-1))
            transforms.append(msc)
        else:
            for i in range(num_blocks):
                block = GlowBlock(shape, hid_channels, num_steps, dim_context)
                transforms.append(block)
                shape = block.output_shape
        super().__init__(transforms)

if __name__ == '__main__':

    glow = Glow((3, 64, 64), 256, 3, 10, dim_context=128)
    trans = glow
    x = torch.randint(0, 256, [10, 3, 64, 64]).float() / 255.

    #trans = GlowBlock((3, 64, 64), 256, 1, dim_context=128)
    #x = torch.randn(10, 3, 64, 64)

    context = torch.randn(10, 128)
    y, _ = trans(x, context)
    xr, _ = trans.inverse(y, context)

    #print(((x-xr)/(x + 1e-10)).mean())
    print(((x-xr).mean()))

    print(x[0][1][0][:10])
    print(xr[0][1][0][:10])
