import torch
import torch.nn as nn


class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(7, 7), stride=2, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(30),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.classify = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        z = self.feature(x)
        z = z.view(-1, 6 * 6 * 256)
        z = self.classify(z)
        return z


if __name__ == "__main__":
    net = ZFNet()
    img = torch.randn(2, 3, 224, 224)
    score = net(img)
    print(score)

