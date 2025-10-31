from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import torch


def t0():
    net = models.densenet121(pretrained=True)
    net.eval().cpu()
    print(net)


def t1():
    net = models.densenet121(pretrained=True)
    net.eval().cpu()

    tfs = transforms.ToTensor()

    image_path = {
        "小狗": r"./datas/小狗.png",
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


if __name__ == '__main__':
    # t0()
    t1()
