import torch
import torch.nn as nn
from pathlib import Path
import os
import torch.nn.functional as F


class AdaptiveAvgPool2dModule(nn.Module):  # 自适应平均池化
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dModule, self).__init__()
        self.k = output_size

    def forward(self, x):
        k = self.k
        n, c, h, w = x.shape
        hk = h // k
        if h % k != 0:
            hk += 1
        wk = w // k
        if w % k != 0:
            wk += 1
        ph = hk * k - h  # 填充大小
        pw = wk * k - h  # 填充大小
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        x = x.reshape(n, c, k, hk, k, wk)
        x = torch.permute(x, dims=(0, 1, 2, 4, 3, 5))
        x = torch.mean(x, dim=(4, 5))
        return x


class VggNet(nn.Module):
    def __init__(self, features, num_classes, classify_input_channel):
        super(VggNet, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.classify_input_channel = classify_input_channel
        self.pooling = AdaptiveAvgPool2dModule(7)  # onnx自适应池化导出错误
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * self.classify_input_channel, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, images):
        x = self.features(images)
        x = self.pooling(x)
        x = x.flatten(1)
        # x = x.view(-1, 7 * 7 * self.classify_input_channel)
        x = self.classify(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, n, use_11=False, use_lrn=False):
        super(Block, self).__init__()
        layers = []
        for i in range(n):
            if (i == n - 1) and use_11:
                kernel_size = (1, 1)
                padding = 0  # onnx导出padding="same"时，报错
            else:
                kernel_size = (3, 3)
                padding = 1
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding),
                nn.ReLU()
            )
            in_channel = out_channel
            layers.append(conv)
        if use_lrn:
            layers.append(nn.LocalResponseNorm(10))
        layers.append(nn.MaxPool2d((2, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Vgg16Net(nn.Module):
    def __init__(self, num_classes):
        super(Vgg16Net, self).__init__()
        self.features = nn.Sequential(
            Block(3, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 3),
            Block(256, 512, 3),
            Block(512, 512, 3)
        )
        self.num_classes = num_classes
        self.vgg = VggNet(self.features, self.num_classes, 512)

    def forward(self, images):
        return self.vgg(images)


class Vgg19Net(nn.Module):
    def __init__(self, num_classes):
        super(Vgg19Net, self).__init__()
        self.features = nn.Sequential(
            Block(3, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 4),
            Block(256, 512, 4),
            Block(512, 512, 4)
        )
        self.num_classes = num_classes
        self.vgg = VggNet(self.features, self.num_classes, 512)

    def forward(self, images):
        return self.vgg(images)


class Vgg16cNet(nn.Module):
    def __init__(self, num_classes):
        super(Vgg16cNet, self).__init__()
        self.features = nn.Sequential(
            Block(3, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 3, use_11=True),
            Block(256, 512, 3, use_11=True),
            Block(512, 512, 3, use_11=True)
        )
        self.num_classes = num_classes
        self.vgg = VggNet(self.features, self.num_classes, 512)

    def forward(self, images):
        return self.vgg(images)


class Vgg11lrnNet(nn.Module):
    def __init__(self, num_classes):
        super(Vgg11lrnNet, self).__init__()
        self.features = nn.Sequential(
            Block(3, 64, 1, use_lrn=True),
            Block(64, 128, 1),
            Block(128, 256, 2),
            Block(256, 512, 2),
            Block(512, 512, 2)
        )
        self.num_classes = num_classes
        self.vgg = VggNet(self.features, self.num_classes, 512)

    def forward(self, images):
        return self.vgg(images)


class VggLabelNet(nn.Module):
    def __init__(self, vgg):
        super(VggLabelNet, self).__init__()
        self.vgg = vgg
        self.idname = {
            0: "狗",
            1: "猫",
            2: "牛",
            3: "羊"
        }

    def forward(self, images):
        scores = self.vgg(images)
        pred_indexs = torch.argmax(scores, dim=1)
        pred_indexs = pred_indexs.detach().numpy()
        result = []
        for idx in pred_indexs:
            result.append(self.idname[idx])
        return result


def export_onnx(net, example):
    net_dir = Path(r"./output/net/vggnet")
    if not net_dir.exists():
        net_dir.mkdir(parents=True)
    torch.onnx.export(
        model=net.eval().cpu(),
        args=example,
        f=net_dir / "vgg16.onnx",
        input_names=["images"],
        output_names=["scores"],
        opset_version=12,
        dynamic_axes={
            "images": {
                0: "N",
                2: "H",
                3: "W"
            },
            "scores": {
                0: "N"
            }
        }
        # dynamic_axes=None
    )


if __name__ == '__main__':
    x = torch.rand(4, 3, 224, 224)
    vgg16 = Vgg16Net(4)
    vgg_label = VggLabelNet(vgg16)
    print(vgg_label)
    r = vgg_label(x)
    print(r)
    export_onnx(vgg16, x)
    # path = "./output/net/vggnet/vgg16.pt"
    # if not os.path.exists(os.path.dirname(path)):
    #     os.makedirs(os.path.dirname(path))
    # traced_script_module = torch.jit.trace(vgg16, x)
    # traced_script_module.save(path)
