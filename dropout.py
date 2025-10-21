import numpy as np
import torch
import torch.nn as nn


# print(np.random.permutation(6))
#
# a = np.array([1, 2, 3, 4, 5, 6])
# b = np.random.permutation(a)
# index = b[0: 3]
# print(index)
# print(a[index])
def t1():
    x = torch.rand(20)
    d1 = nn.Dropout(0.6)
    x3 = x / (1 - 0.6)
    d1.train()
    x1 = d1(x)
    d1.eval()
    x2 = d1(x)
    print(x)
    print(x1)
    print(x2)
    print(x3)


def t2():
    x = torch.rand(1, 10, 2, 2)
    d2 = nn.Dropout2d(0.5)
    d2.train()
    x1 = d2(x)
    d2.eval()
    x2 = d2(x)
    print(x)
    print(x1)
    print(x2)


if __name__ == '__main__':
    # t1()
    t2()
