import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)
