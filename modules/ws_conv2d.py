import torch
import torch.nn as nn


class WSConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 device=None,
                 dtype=None,
                 eps=1e-8):
        super(WSConv2d, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups,
                                       bias,
                                       padding_mode,
                                       device,
                                       dtype)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((out_channels, 1, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((out_channels, 1, 1, 1)))

    def forward(self, x):
        if self.training:
            w = self.weight
            w_mean = torch.mean(w, dim=(1, 2, 3), keepdim=True)
            w_var = torch.var(w, dim=(1, 2, 3), keepdim=True)
            w = (w - w_mean) / torch.sqrt(w_var + self.eps)
            w = w * self.gamma + self.beta
            self.weight.data = w
            return super(WSConv2d, self).forward(x)


if __name__ == '__main__':
    conv = WSConv2d(3, 64, (3, 3), 1, 1)
    _x = torch.rand(4, 3, 28, 28)
    _r = conv(_x)
    print(_r.shape)
