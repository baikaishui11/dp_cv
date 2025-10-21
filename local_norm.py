import torch
import torch.nn as nn

if __name__ == '__main__':
    norm = nn.LocalResponseNorm(size=5)
    x = torch.randn(2, 32, 128, 128)
    x = norm(x)
    print(x.shape)
