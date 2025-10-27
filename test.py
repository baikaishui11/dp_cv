import torch
import torch.nn as nn
from torchvision import models
import torch


if __name__ == '__main__':
    # pool = nn.MaxPool2d(3, stride=2)
    # x = torch.randn(2, 48, 13, 13)
    # x = pool(x)
    # print(x.shape)
    # net = models.AlexNet()
    # print(net)
    # z = torch.flatten(x, 2)
    # print(z.shape)
    # lst = [1, 2, 3, 4]
    # print(lst.val)
    # x = torch.rand((3, 4, 55, 55))
    # n, c, h, w = x.shape
    # for i in range(n):
    #     a = x[i: i+1]
    #     print(a.shape)
    model = models.googlenet(pretrained=True)
    print(model)