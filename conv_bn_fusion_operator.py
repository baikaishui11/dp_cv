from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(55)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            Conv(3, 64, 3, 1, 1),
            Conv(64, 128, 3, 2, 1),  # 下采样
            Conv(128, 128, 3, 1, 1),
            Conv(128, 256, 3, 2, 1),  # 下采样
            Conv(256, 256, 3, 1, 1),
            nn.AdaptiveMaxPool2d((4, 4))
        )
        self.classify = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        z = self.features(x)
        z = z.flatten(1)
        z = self.classify(z)
        return z


def t0():
    net = Network(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    train_op = optim.SGD(net.parameters(), lr=0.0001)

    n = 10
    xs = [torch.rand(8, 3, 28, 28) for _ in range(n)]
    ys = [torch.randint(10, size=(8,)) for _ in range(n)]
    for epoch in range(2):
        for i in range(n):
            _x = xs[i]
            _y = ys[i]
            loss = loss_fn(net(_x), _y)

            train_op.zero_grad()
            loss.backward()
            train_op.step()

            print(f"epoch:{epoch}, batch:{i}, loss:{loss.item():.5f}")

    path_dir = Path("./output/models/01")
    path_dir.mkdir(parents=True, exist_ok=True)

    torch.save(net.eval(), str(path_dir / "model.pkl"))


def export(model_dir, model_path=None, name="model"):
    model_dir = Path(model_dir)

    if model_path is None:
        model_path = model_dir / "model.pkl"
    net = torch.load(model_path, map_location="cpu", weights_only=False)
    net.eval().cpu()

    example = torch.rand(1, 3, 32, 32)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(model_dir / f"{name}.pt")

    torch.onnx.export(
        model=net,
        args=example,
        f=model_dir / "model_dynamic.onnx",
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={
            "images": {
                0: "batch"
            },
            "scores": {
                0: "batch"
            }
        }
    )


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    fuseconv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    )
    fuseconv.requires_grad_(False).to(conv.weight.device)
    w_bn = bn.weight.div(torch.sqrt(bn.running_var + bn.eps))
    w_bn_conv = w_bn[:, None, None, None]  # OC --> OC,1,1,1
    new_weight = conv.weight.clone() * w_bn_conv
    fuseconv.weight.copy_(new_weight)
    conv_bias = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias.clone()
    new_bias = (conv_bias - bn.running_mean) * w_bn + bn.bias
    fuseconv.bias.copy_(new_bias)
    return fuseconv


def fuse_modules(model_dir, name="new_model"):
    model_dir = Path(model_dir)

    net = torch.load(model_dir / "model.pkl", map_location="cpu", weights_only=False)
    net.eval().cpu()
    for m in net.modules():
        if type(m) is Conv:
            # 进行模块合并（conv 和 bn）
            m.conv = fuse_conv_bn(m.conv, m.bn)
            delattr(m, "bn")  # 删除m对象中的bn属性
            m.forward = m.fuse_forward  # 方法的赋值
    torch.save(net.cpu(), str(model_dir / f"{name}.pkl"))

    export(model_dir=model_dir, model_path=str(model_dir / f"{name}.pkl"), name=name)


def tt_fuse(model_dir):
    model_dir = Path(model_dir)

    net1 = torch.jit.load(str(model_dir / "model.pt"), map_location="cpu")
    net2 = torch.jit.load(str(model_dir / "new_model.pt"), map_location="cpu")

    x = torch.rand(4, 3, 28, 28)
    r1 = net1(x)
    r2 = net2(x)
    print(r1 - r2)


def tt(model_dir):
    model_dir = Path(model_dir)

    net = torch.load(model_dir / "model.pkl", map_location="cpu", weights_only=False)
    net.eval().cpu()

    # 直接调用torch量化接口
    # modules_to_fuse:网络结构中的属性名称可以参考net.state_dict()的key来给定
    fused_m = torch.quantization.fuse_modules(model=net,
                                              modules_to_fuse=[
                                                  ["features.0.conv", "features.0.bn", "features.0.act"],
                                                  ["features.1.conv", "features.1.bn", "features.1.act"],
                                                  ["features.2.conv", "features.2.bn", "features.2.act"],
                                                  ["features.3.conv", "features.3.bn", "features.3.act"],
                                                  ["features.4.conv", "features.4.bn", "features.4.act"],
                                              ])

    print(fused_m)
    x = torch.rand(4, 3, 28, 28)
    r1 = net(x)
    r2 = fused_m(x)
    print(r1 - r2)

    torch.save(fused_m.cpu(), str(model_dir / "fuse_model.pkl"))

    export(model_dir=model_dir, model_path=str(model_dir / "fuse_model.pkl"), name="fuse_model")


if __name__ == '__main__':
    # t0()
    # fuse_modules("./output/models/01")
    # export("./output/models/01")
    # tt_fuse("./output/models/01")
    tt("./output/models/01")
