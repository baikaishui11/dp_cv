import torch
import torch.nn as nn


if __name__ == "__main__":
    alpha = nn.Parameter(torch.empty(3, 5))
    nn.init.kaiming_uniform_(alpha, mode="fan_out", nonlinearity="sigmoid")
    nn.init.ka
    print(alpha)