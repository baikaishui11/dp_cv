import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, score, target):
        pre_inex = torch.argmax(score, dim=1).to(device=target.device, dtype=target.dtype)
        corr = (pre_inex == target).to(dtype=torch.float)
        acc = torch.mean(corr)
        return corr.shape[0], acc



