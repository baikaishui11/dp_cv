import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgPool2dModule(nn.Module):  # 自适应平均池化
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dModule, self).__init__()
        self.k = output_size

    def forward(self, x):
        k = self.k
        n, c, h, w = x.shape
        hk = h // k
        if h % k != 0:
            hk += 1
        wk = w // k
        if w % k != 0:
            wk += 1
        ph = hk * k - h  # 填充大小
        pw = wk * k - h  # 填充大小
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        x = x.reshape(n, c, k, hk, k, wk)
        x = torch.permute(x, dims=(0, 1, 2, 4, 3, 5))
        x = torch.mean(x, dim=(4, 5))
        return x
