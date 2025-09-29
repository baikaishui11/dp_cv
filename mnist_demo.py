import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    dataset = datasets.MNIST(root="./datas/MNIST",
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True
                             )

    data_loader = DataLoader(dataset, batch_size=8)
    k = 0
    for batch_img, batch_label in data_loader:
        n, c, h, w = batch_img.shape
        for i in range(n):
            img = batch_img[i].detach().numpy()
            gray_img = (img[0]*256).astype(np.uint8)
            label = batch_label[i].item()
            output_path = f"./datas/MNIST/MNIST/{label}/{k}.png"
            k += 1
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            plt.imsave(output_path, gray_img, cmap="gray")
