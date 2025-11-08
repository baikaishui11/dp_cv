from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class SN(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        super(SN, self).__init__()
        # register_buffer 将属性当成parameter进行处理，唯一的区别是不进行反向传播的梯度求解
        self.register_buffer("running_mean", torch.zeros((1, num_features, 1, 1)))
        self.register_buffer("running_var", torch.zeros((1, num_features, 1, 1)))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]

        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        self.momentum = momentum
        self.eps = eps
        self.w = nn.Parameter(torch.ones(3))

    def get_bn(self, x):
        _mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        _var = torch.var(x, dim=(0, 2, 3), keepdim=True)
        if self.training:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * _mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * _var
        else:
            _mean = self.running_mean
            _var = self.running_var
        return _mean, _var

    def get_ln(self, x):
        _mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        _var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        return _mean, _var

    def get_in(self, x):
        _mean = torch.mean(x, dim=(2, 3), keepdim=True)
        _var = torch.var(x, dim=(2, 3), keepdim=True)
        return _mean, _var

    def forward(self, x):
        _bn_mean, _bn_var = self.get_bn(x)
        _ln_mean, _ln_var = self.get_ln(x)
        _in_mean, _in_var = self.get_in(x)

        w = torch.softmax(self.w, dim=0)
        bn_w, ln_w, in_w = w[0], w[1], w[2]

        _mean = _bn_mean * bn_w + _ln_mean * ln_w + _in_mean * in_w
        _var = _bn_var * bn_w + _ln_var * ln_w + _in_var * in_w

        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


def t0():
    path_dir = Path("../output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    net = SN(num_features=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.train()
    xs = [torch.rand(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        net(_x)
    print(net.running_mean.view(-1))
    print(net.running_var.view(-1))

    net.eval()
    _r = net(xs[0])
    print(_r.shape)

    net = net.cpu()
    torch.save(net, str(path_dir / "sn_model.pkl"))
    torch.save(net.state_dict(), str(path_dir / "sn_params.pkl"))

    traced_script_module = torch.jit.trace(net, xs[0].cpu())
    traced_script_module.save(str(path_dir / "sn_model.pt"))


if __name__ == '__main__':
    t0()
