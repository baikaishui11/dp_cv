from PIL import Image
from torchvision import transforms
import numpy as np
import torch


def t1():
    img = Image.open(r"../datas/小狗.png").convert("RGB")
    ts0 = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.RandomPosterize(bits=5, p=1.0)

    ])
    print(np.unique(ts0(img).data.numpy()))
    transforms.ToPILImage()(ts0(img)).show()


if __name__ == '__main__':
    t1()
