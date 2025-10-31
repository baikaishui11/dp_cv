import onnx
import torch
import torch.nn as nn
import torch.serialization


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(BasicConv2d, self).__init__()
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#
#     def forward(self, x):
#         return self.relu(self.conv(self.bn(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:
        :param out_channels: 各个分支的输出[[64], [96, 128], [16, 32], [32]]
        """
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[0][0], kernel_size=(1, 1), stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[1][0], kernel_size=(1, 1), stride=1, padding=0),
            BasicConv2d(out_channels[1][0], out_channels[1][1], kernel_size=(3, 3), stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[2][0], kernel_size=(1, 1), stride=1, padding=0),
            BasicConv2d(out_channels[2][0], out_channels[2][1], kernel_size=(5, 5), stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=1, padding=1),
            BasicConv2d(in_channels, out_channels[3][0], kernel_size=(1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, add_aux_stage=False):
        super(GoogLeNet, self).__init__()
        self.stage1 = nn.Sequential(
            BasicConv2d(3, 64, (7, 7), 2, 3),
            nn.MaxPool2d((3, 3), 2, 1),
            BasicConv2d(64, 64, (1, 1), 1, 0),
            BasicConv2d(64, 192, (3, 3), 1, 1),
            nn.MaxPool2d((3, 3), 2, 1),
            Inception(192, [[64], [96, 128], [16, 32], [32]]),  # inception3a
            Inception(256, [[128], [128, 192], [32, 96], [64]]),  # inception3b
            nn.MaxPool2d((3, 3), 2, 1),
            Inception(480, [[192], [96, 208], [16, 48], [64]])
            )
        self.stage2 = nn.Sequential(
            Inception(512, [[160], [112, 224], [24, 64], [64]]),
            Inception(512, [[128], [128, 256], [24, 64], [64]]),
            Inception(512, [[112], [144, 288], [32, 64], [64]])
        )
        self.stage3 = nn.Sequential(
            Inception(528, [[256], [160, 320], [32, 128], [128]]),
            nn.MaxPool2d((2, 2), 2, 1),
            Inception(832, [[256], [160, 320], [32, 128], [128]]),
            Inception(832, [[384], [192, 384], [48, 128], [128]]),
            GlobalAvgPool2d()
        )
        self.classify = nn.Conv2d(1024, num_classes, (1, 1))
        if add_aux_stage:
            self.aux_stage1 = nn.Sequential(
                nn.MaxPool2d((5, 5), 3),
                nn.Conv2d(512, 1024, (1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_classes)
            )
            self.aux_stage2 = nn.Sequential(
                nn.MaxPool2d((5, 5), 3),
                nn.Conv2d(528, 1024, (1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_classes)
            )

    def forward(self, x):
        z1 = self.stage1(x)
        z2 = self.stage2(z1)
        z3 = self.stage3(z2)

        scores3 = torch.squeeze(self.classify(z3))
        if self.aux_stage1 is not None:
            scores1 = self.aux_stage1(z1)
            scores2 = self.aux_stage2(z2)
            return scores1, scores2, scores3
        else:
            return scores3


def t1():
    net = GoogLeNet(4, add_aux_stage=True)
    _x = torch.rand(2, 3, 224, 224)
    _r1, _r2, _r3 = net(_x)
    print(_r1)
    print(_r2)
    print(_r3)

    net.aux_stage1 = None
    net.aux_stage2 = None

    traced_script_module = torch.jit.trace(net.eval(), _x)
    traced_script_module.save("./output/models/googlenet_aux.pt")
    torch.save(net, "./output/models/googlenet.pkl")
    torch.onnx.export(
        model=net.eval().cpu(),
        args=_x,
        f="./output/models/googlenet.onnx",
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
    )


def t2():
    net1 = torch.load("./output/models/googlenet.pkl", map_location="cpu", weights_only=False)
    net2 = GoogLeNet(4, add_aux_stage=False)
    # missing_keys: 表示net2中有部分参数没有恢复
    # unexpected_keys: 表示net2中没有这部分参数， 但是入参的字典中传入了该参数
    missing_keys, unexpected_keys = net2.load_state_dict(net1.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise ValueError(f"网络有部分参数没有恢复:{missing_keys}")
    print(unexpected_keys)

    _x = torch.rand(2, 3, 224, 224)
    traced_script_module = torch.jit.trace(net2.eval(), _x)
    traced_script_module.save("./output/models/googlenet.pt")

    # torch.onnx.export(
    #     model=net2.eval().cpu(),
    #     args=_x,
    #     f="./output/models/googlenet.onnx",
    #     input_names=["images"],
    #     output_names=["scores"],
    #     opset_version=12,
    #     dynamic_axes={
    #         "images": {
    #             0: "N",
    #             2: "H",
    #             3: "W"
    #         },
    #         "scores": {
    #             0: "N"
    #         }
    #     }
    # )


if __name__ == '__main__':
    t1()
    # t2()
