import torch
import torch.nn as nn
from torchvision import models


class FastRCNN(nn.Module):
    def __init__(self):
        super(FastRCNN, self).__init__()
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.features = self.vgg.features
        del self.features[43]

        self.features2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(1),
            nn.Linear(512 * 4 * 4, 16),
            nn.Linear(16, 4096),
            nn.ReLU(),
            nn.Linear(4096, 32),
            nn.Linear(32, 4096),
            nn.ReLU()
        )
        self.classify_header = nn.Linear(4096, 21)
        self.reg_header = nn.Linear(4096, 4)

    def forward(self, images, roi_list):
        features = self.features(images)
        roi_classify_sources = []
        roi_reg_sources = []
        for image_idx, img_roi in enumerate(roi_list):
            img_feature = features[image_idx: image_idx + 1]
            for lx, ly, rx, ry in img_roi:
                roi_feature = img_feature[:, :, ly:ry, lx:rx]
                roi_feature = self.features2(roi_feature)

                roi_classify = self.classify_header(roi_feature)
                roi_classify_sources.append(roi_classify)

                roi_reg = self.reg_header(roi_feature)
                roi_reg_sources.append(roi_reg)

        roi_classify_sources = torch.concat(roi_classify_sources, dim=0)
        roi_reg_sources = torch.concat(roi_reg_sources, dim=0)
        return roi_classify_sources, roi_reg_sources


if __name__ == '__main__':
    net = FastRCNN()
    print(net.features)

    cla_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")

    img = torch.rand(2, 3, 224, 224)
    roi_list = [
        [
            (1, 0, 5, 2),
            (2, 3, 8, 9),
            (3, 4, 10, 12),
            (5, 9, 12, 13)
        ],
        [
            (4, 0, 6, 2),
            (2, 2, 8, 6),
            (3, 4, 7, 12),
            (5, 3, 12, 13)
        ]
    ]
    roi_targets_labels = torch.tensor([0, 0, 0, 2, 0, 5, 0, 0])
    roi_targets_reg = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.3, -0.2, 0.8, 1.2],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.9, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    # _r = net.features(img)
    # print(_r.shape)
    _r1, _r2 = net(img, roi_list)
    print(_r1.shape)
    print(_r2.shape)

    _cla_loss = cla_loss_fn(_r1, roi_targets_labels)
    _reg_loss = torch.mean(torch.sum(reg_loss_fn(_r2, roi_targets_reg), dim=1) * (roi_targets_labels >= 1).to(roi_targets_reg.dtype))

    print(_cla_loss)
    print(_reg_loss)
