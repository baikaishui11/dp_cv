import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    dataset = datasets.CIFAR100(root="./datas/CIFAR100",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True
                                )

    data_loader = DataLoader(dataset, batch_size=8)
    k = 0
    for batch_img, batch_label in data_loader:
        n, c, h, w = batch_img.shape  # N,C,H,W
        for i in range(n):
            img = torch.permute(batch_img[i], dims=(1, 2, 0)).detach().numpy()  # C,H,W --> H,W,C
            gray_img = (img * 256).astype(np.uint8)
            label = batch_label[i].item()
            output_path = f"./datas/CIFAR100/CIFAR100/{label}/{k}.png"
            k += 1
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            plt.imsave(output_path, gray_img)
