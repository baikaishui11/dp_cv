import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from torchvision import transforms


class DropBlock2d(nn.Module):
    def __init__(self, p: float = 0.1, block_size: int = 7, inplace: bool = False):
        super(DropBlock2d, self).__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        if block_size < 1:
            raise ValueError("block_size必须大于等于1")

    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        N, C, H, W = input.size()

        mask_h = H - self.block_size + 1
        mask_w = W - self.block_size + 1
        gama = (self.p * H * W) / ((self.block_size ** 2) * mask_h * mask_w)
        mask_shape = (N, C, mask_h, mask_w)
        mask = torch.bernoulli(torch.full(mask_shape, gama, device=input.device))
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        mask = F.max_pool2d(mask, (self.block_size, self.block_size), (1, 1), self.block_size // 2)
        mask = 1.0 - mask

        normalize_scale = mask.numel() / (1e-6 + mask.sum())

        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale

        return input


def t1():
    feature = torch.rand((2, 3, 20, 20))

    dropout = nn.Dropout(p=0.1)
    dropout_feature = dropout(feature)

    print(dropout_feature)

    dropblock = DropBlock2d(p=0.1)
    dropblock_feature = dropblock(feature)

    print(dropblock_feature)


def t2():
    img = Image.open(r"../datas/小狗.png").convert("L")
    ts1 = transforms.ToTensor()
    ts2 = transforms.ToPILImage()
    img = ts1(img)[None]

    p = 0.2
    dropout = nn.Dropout(p=p)
    dropblock = DropBlock2d(p=p)
    img1 = ts2(dropout(img)[0])
    img2 = ts2(dropblock(img)[0])

    img1.show("img1")
    img2.show("img2")


if __name__ == '__main__':
    # t1()
    t2()
