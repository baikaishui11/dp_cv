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
    a = torch.rand(16, 28, 28)
    c, h, w = a.shape
    for i in range(c):
        b = a[i: i+1]
        print(b.shape)
        print(b)