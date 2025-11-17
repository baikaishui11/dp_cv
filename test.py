import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
# class A():
#     def __init__(self, x):
#         self.x = x
#         x = 10
#
#     def b(self):
#         return self.x
from conv_demo_mnist import Network

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
    # model = models.googlenet(pretrained=True)
    # print(model)
    # r = A(5)
    # print(r.b())
    # x = torch.tensor([[1, 2],
    #                   [3, 4]])
    #
    # # 填充规则：左1，右1，上0，下0（即左右各补1行0）
    # pad = (1, 2, 3, 0)
    # y = F.pad(x, pad, mode='constant', value=0)
    #
    # print(y)
    # print(x.shape)
    # print(y.shape)
    # net = Network(in_channels=1, num_classes=10, img_h=28, img_w=28)
    # print(net)
    # a = torch.rand((1, 3, 28, 28))
    # b = torch.squeeze(a, dim=0)
    # print(a.shape)
    # print(b.shape)
    # a = torch.rand(16, 28, 28)
    # c, h, w = a.shape
    # for i in range(c):
    #     b = a[i: i+1]
    #     print(b.shape)
    #     print(b)
    # a = torch.rand(4, 150)
    # mean = torch.mean(a, dim=1, keepdim=True)
    # std = torch.std(a, dim=1, keepdim=True)
    # z = mean / (std + 1e-5)
    # w = torch.rand(4, 128)
    # x = w * z
    #
    # print(mean.shape)
    # print(std.shape)
    # print(z.shape)
    # print(x.shape)
    # a = torch.rand(8, 12, 32, 32)
    # mean = torch.mean(a, dim=(1, 2, 3), keepdim=True)
    # std = torch.std(a, dim=(1, 2, 3), keepdim=True)
    # # z = mean / (std + 1e-5)
    # # w = torch.rand(4, 128)
    # # x = w * z
    #
    # print(mean.shape)
    # print(std.shape)
    # a = torch.rand(16)
    # print(a.shape)
    # a = a[:, None, None, None]
    # print(a.shape)
    # convbn = nn.Conv2d(8, 8, 1, 1)
    # nw = torch.ones(8)
    # nw = torch.diag(nw)
    # nw = nw[:, :, None, None]
    # print(nw.shape)
    # import numpy as np
    #
    # a = np.array([1, 5, 2, 3, 2, 3, 5, 2])
    # # b = np.bincount(a)
    # # print(b)
    # b = np.array([1, 5, 2, 3, 2, 2, 5, 2])
    #
    # c = np.bincount(a, b)
    # print(c)
    # print(c.shape)
    from torchvision import datasets
    
