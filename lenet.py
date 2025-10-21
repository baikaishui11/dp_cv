import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            nn.AdaptiveMaxPool2d(output_size=(4, 4))
        )
        self.classify = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        z = self.feature(x)
        z = z.view(-1, 800)
        z = self.classify(z)
        return z


if __name__ == "__main__":
    net = LeNet()
    img = torch.randn(2, 1, 60, 60)
    score = net(img)
    probs = torch.softmax(score, dim=1)
    print(score)
    print(probs)
