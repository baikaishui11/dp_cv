import torch
import torch.nn as nn
import torch.nn.functional as F

torch.random.manual_seed(11)


def t1():
    x = torch.tensor([[1, 2.0, 3], [4, 5, 6]])
    w = nn.Parameter(torch.rand(3, 4))
    bn = nn.BatchNorm1d(4)
    z = x @ w
    z = bn(z)
    o = F.sigmoid(z)
    print(o)


def t2():
    x = torch.randn(8, 32, 128, 126) * 0.1 + 5
    bn = nn.BatchNorm2d(32)
    print(F.sigmoid(x))
    print(F.sigmoid(bn(x)))


def t3():
    z = torch.randn(8, 32, 128, 126) * 0.1 + 5
    # bn_mean = torch.mean(z, dim=(0, 2, 3), keepdim=True)
    bn_norm = nn.BatchNorm2d(32, momentum=1)
    # ln_mean = torch.mean(z, dim=(1, 2, 3), keepdim=True)
    # ln_norm = nn.LayerNorm([1, 2, 3])
    # in_mean = torch.mean(z, dim=(2, 3), keepdim=True)
    # in_norm = nn.InstanceNorm2d(32)
    # print(bn_mean.shape)
    # print(ln_mean.shape)
    # print(in_mean.shape)

    # gz = z.reshape(8, 2, 16, 128, 126)
    # gn_mean = torch.mean(gz, dim=(2, 3, 4), keepdim=True)
    # gn_norm = nn.GroupNorm(2, 32)
    # print(gn_mean.shape)
    print(bn_norm(z))

class bn_demo(nn.Module):
    def __init__(self):
        super(bn_demo, self).__init__()

    def forward(self, x):
        self.mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        self.std = torch.std(x, dim=(0, 2, 3), keepdim=True)
        res = (x - self.mean) / (self.std + 1e-5)
        return res


def t4():
    z = torch.randn(8, 32, 128, 126) * 0.1 + 5
    bn_norm1 = bn_demo()
    print(bn_norm1(z))


if __name__ == "__main__":
    # t1()
    # t2()
    t3()
    t4()