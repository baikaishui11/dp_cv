import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.model(x)


class RegL1loss(nn.Module):
    def __init__(self, lam, params):
        super(RegL1loss, self).__init__()
        self.lam = lam
        self.params = list(params)
        self.n_params = len(self.params)

    def forward(self):
        ls = 0.0
        for param in self.params:
            ls += torch.sum(torch.abs(param))
        return self.lam * ls / self.n_params


if __name__ == "__main__":
    net = Network()
    reg_l1_loss_fn = RegL1loss(0.1, net.parameters())
    _l1_loss = reg_l1_loss_fn()
    print(_l1_loss)
