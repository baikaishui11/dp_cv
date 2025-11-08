from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class LN(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(LN, self).__init__()

        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        self.eps = eps

    def forward(self, x):
        _mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        _var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


def t0():
    path_dir = Path("../output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    ln = LN(num_features=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ln.to(device)

    xs = [torch.rand(8, 12, 32, 32).to(device) for _ in range(10)]

    ln.eval()
    _r = ln(xs[0])
    print(_r.shape)

    ln = ln.cpu()
    torch.save(ln, str(path_dir / "ln_model.pkl"))
    torch.save(ln.state_dict(), str(path_dir / "ln_params.pkl"))

    traced_script_module = torch.jit.trace(ln, xs[0].cpu())
    traced_script_module.save(str(path_dir / "ln_model.pt"))

    ln_model = torch.load(str(path_dir / "ln_model.pkl"), map_location="cpu", weights_only=False)
    ln_params = torch.load(str(path_dir / "ln_params.pkl"), map_location="cpu", weights_only=False)

    print(len(ln_params))


if __name__ == '__main__':
    t0()
