import torch.nn as nn
from torchvision import models
import torch


class VggNet(nn.Module):
    def __init__(self, num_classes):
        super(VggNet, self).__init__()

        # self.vgg = models.vgg16_bn(pretrained=True)
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        del self.vgg.classifier[6]

        self.classify = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        z = self.vgg(x)
        return self.classify(z)

    def feature_forward(self, x):
        z1 = self.vgg.features(x)
        z2 = self.vgg.avgpool(z1)
        z2 = torch.flatten(z2, 1)
        z3 = self.vgg.classifier(z2)
        return z2, z3


if __name__ == '__main__':
    vgg = VggNet(num_classes=21)
    print(vgg)
    _x = torch.rand(2, 3, 224, 224)
    _r = vgg(_x)
    print(_r.shape)

    _r5, _r7 = vgg.feature_forward(_x)
    print(_r5.shape)
    print(_r7.shape)



