import time
from PIL import Image
from torchvision import models, transforms
import torch
from pathlib import Path
from thop import profile
import torch.nn as nn


def t0():
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    net = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    _x = torch.rand(4, 3, 224, 224)
    shufflenet_v2_model = torch.jit.trace(net.eval(), _x)
    shufflenet_v2_model.save(path_dir / "shufflenet_v2.pt")
    print(net)
    print("=" * 100)



def t1():
    net = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    net.eval().cpu()

    tfs = transforms.ToTensor()

    image_path = {
        "小狗": r"./datas/dog.png",
        "小狗2": r"./datas/小狗2.png",
        "小猫": r"./datas/小猫.jpg",
        "小猫2": r"./datas/小猫2.jpg",
        "飞机": r"./datas/飞机.jpg",
        "飞机2": r"./datas/飞机2.jpg",
    }
    for name in image_path.keys():
        print("==" * 50)
        img = Image.open(image_path[name]).convert("RGB")  # 加载图像并将图像转换为RGB3通道
        img = tfs(img)
        img = img[None]  # 增加维度从CHW-->1CHW

        scores = net(img)
        pre_index = torch.argmax(scores, dim=1)
        prob = torch.softmax(scores, dim=1)
        top5 = torch.topk(prob, k=5, dim=1)
        print(name)
        print(top5)


def calc_flops(net, inputs):
    print(type(net))
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    elif not isinstance(inputs, tuple):
        inputs = (inputs,)
    flops, params = profile(net, inputs)
    print(f"总的浮点运算量{flops}")
    print(f"总的参数量{params}")

    net.eval()
    with torch.no_grad():
        start_time = time.time()
        net(*inputs)
        end_time = time.time()
        print(f"耗时:{end_time - start_time}")


def t2():
    print("=" * 100)
    calc_flops(net=nn.Sequential(nn.Linear(3, 5)), inputs=torch.rand(2, 3))
    print("=" * 100)
    _x = torch.randn(1, 3, 224, 224)
    vgg = models.vgg16_bn(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.resnet101(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.densenet121(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.mobilenet_v2(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.mobilenet_v3_small(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.mobilenet_v3_large(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.shufflenet_v2_x0_5(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)
    vgg = models.shufflenet_v2_x2_0(weights=None)
    calc_flops(vgg, _x)
    print("=" * 100)


if __name__ == '__main__':
    t0()
    t1()
    t2()
