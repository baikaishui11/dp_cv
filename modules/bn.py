from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        super(BatchNorm, self).__init__()
        # register_buffer 将属性当成parameter进行处理，唯一的区别是不进行反向传播的梯度求解
        self.register_buffer("running_mean", torch.zeros((1, num_features, 1, 1)))
        self.register_buffer("running_var", torch.zeros((1, num_features, 1, 1)))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]

        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        _mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        _var = torch.var(x, dim=(0, 2, 3), keepdim=True)
        if self.training:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * _mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * _var
        else:
            _mean = self.running_mean
            _var = self.running_var

        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


def t0():
    path_dir = Path("../output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    bn = BatchNorm(num_features=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bn.to(device)

    bn.train()
    xs = [torch.rand(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        bn(_x)
    print(bn.running_mean.view(-1))
    print(bn.running_var.view(-1))

    bn.eval()
    _r = bn(xs[0])
    print(_r.shape)

    bn = bn.cpu()
    torch.save(bn, str(path_dir / "bn_model.pkl"))
    torch.save(bn.state_dict(), str(path_dir / "bn_params.pkl"))

    traced_script_module = torch.jit.trace(bn, xs[0].cpu())
    traced_script_module.save(str(path_dir / "bn_model.pt"))

    bn_model = torch.load(str(path_dir / "bn_model.pkl"), map_location="cpu", weights_only=False)
    bn_params = torch.load(str(path_dir / "bn_params.pkl"), map_location="cpu", weights_only=False)

    print(len(bn_params))


if __name__ == '__main__':
    t0()
