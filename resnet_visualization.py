from typing import Union, Optional, List
from torchvision import models, transforms
from PIL import Image
import torch
import torchvision
from pathlib import Path


class ResNetHook(object):
    def __init__(self, net, names: Optional[List[str]] = None):
        self.hooks = []
        self.images = {}
        if names is None:
            names = [
                'conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'
            ]
        for name in names:
            # 注册一个钩子
            hook = getattr(net, name).register_forward_hook(self._build_hook(name))
            self.hooks.append(hook)

    def _build_hook(self, name):
        def hook(module, module_input, module_output):
            self.images[name] = module_output
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)  # image_net数据集1000类
    # https://blog.csdn.net/winycg/article/details/101722445
    print(model)
    resnet_hooks = ResNetHook(model)
    tfs = transforms.ToTensor()
    resize = transforms.Resize(size=(50, 60))
    model.eval().cpu()
    output_dir = Path("./output/resnet/features/")
    image_path = {
        "小狗": r"./datas/小狗.png",
        "小狗2": r"./datas/小狗2.png",
        "小猫": r"./datas/小猫.jpg",
        "小猫2": r"./datas/小猫2.jpg",
        "飞机": r"./datas/飞机.jpg",
        "飞机2": r"./datas/飞机2.jpg",
    }
    for name in image_path.keys():
        print("=="*50)
        img = Image.open(image_path[name]).convert("RGB")  # 加载图像并将图像转换为RGB3通道
        img = tfs(img)
        img = img[None]  # 增加维度从CHW-->NCHW

        scores = model(img)
        pre_index = torch.argmax(scores, dim=1)
        prob = torch.softmax(scores, dim=1)
        top5 = torch.topk(prob, k=5, dim=1)
        print(name)
        print(top5)

        _output_dir = output_dir / name
        _output_dir.mkdir(parents=True, exist_ok=True)

        for layer_name in resnet_hooks.images.keys():
            features = resnet_hooks.images[layer_name]
            n, c, h, w = features.shape
            for i in range(n):
                imgs = features[i: i+1]  # NCHW --> 1CHW
                imgs = torch.permute(imgs, (1, 0, 2, 3))  # 1CHW --> C1HW
                imgs = resize(imgs)
                torchvision.utils.save_image(
                    imgs,
                    _output_dir / f"{i}_{layer_name}.png",
                    nrow=8,
                    padding=5
                )
    resnet_hooks.remove()
