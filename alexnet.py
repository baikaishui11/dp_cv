import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, device1, device2):
        super(AlexNet, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.feature11 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(48, 128, kernel_size=(5, 5), padding="same"),
            nn.LocalResponseNorm(30),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        ).to(self.device1)
        self.feature21 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(10),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(48, 128, kernel_size=(5, 5), padding="same"),
            nn.LocalResponseNorm(30),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        ).to(self.device2)
        self.feature12 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        ).to(self.device1)
        self.feature22 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=2)
        ).to(self.device2)
        self.classify = nn.Sequential(
            nn.Linear(6 * 6 * 128 * 2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000)
        ).to(self.device1)

    def forward(self, x):
        x1 = x.to(self.device1)
        x2 = x.to(self.device2)

        oz1 = self.feature11(x1)
        oz2 = self.feature21(x2)

        z1 = torch.concatenate([oz1, oz2.to(self.device1)], dim=1)
        z2 = torch.concatenate([oz1.to(self.device2), oz2], dim=1)

        z1 = self.feature12(z1)
        z2 = self.feature22(z2)

        z = torch.concatenate([z1, z2.to(self.device1)], dim=1)
        z = z.view(-1, 6 * 6 * 128 * 2)

        z = self.classify(z)
        return z


if __name__ == '__main__':
    device1 = torch.device("cpu")
    device2 = torch.device("cuda:0")
    net = AlexNet(device1, device2)
    img = torch.randn(2, 3, 224, 224)
    scores = net(img)
    print(scores)
