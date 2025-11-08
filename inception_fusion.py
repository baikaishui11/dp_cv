import torch
import torch.nn as nn
import torch.nn.functional as F


def t0():
    _x = torch.rand(4, 9, 24, 24)

    conv1x1 = nn.Conv2d(9, 9, (1, 1), 1, 0)
    conv3x3 = nn.Conv2d(9, 9, (3, 3), 1, 1)
    r1 = conv1x1(_x) + conv3x3(_x)
    print(r1.shape)

    conv = nn.Conv2d(9, 9, (3, 3), 1, 1).requires_grad_(False)
    conv1x1_weight = F.pad(conv1x1.weight.clone(), [1, 1, 1, 1])
    conv1x1_bias = conv1x1.bias
    conv3x3_weight = conv3x3.weight
    conv3x3_bias = conv3x3.bias
    conv.weight.copy_(conv1x1_weight + conv3x3_weight)
    conv.bias.copy_(conv1x1_bias + conv3x3_bias)

    r2 = conv(_x)
    print(r2.shape)

    r = torch.abs(r1 - r2)
    print(torch.max(r))


if __name__ == '__main__':
    t0()
