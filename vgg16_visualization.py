from typing import Union
from torchvision import models, transforms
from PIL import Image
import torch
import torchvision
from pathlib import Path


class VggHook(object):
    def __init__(self, vgg, indexes: Union[int, list[int]] = 44):
        self.hooks = []
        self.images = {}
        if isinstance(indexes, int):
            indexes = list(range(indexes))
        for idx in indexes:
            # 注册一个钩子
            self.hooks.append(vgg.features[idx].register_forward_hook(self._build_hook(idx)))

    def _build_hook(self, idx):
        def hook(module, module_input, module_output):
            self.images[idx] = module_output
        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()


if __name__ == '__main__':
    vgg = models.vgg16_bn(pretrained=True)  # image_net数据集1000类
    # https://blog.csdn.net/winycg/article/details/101722445
    # print(vgg)
    vgg_hooks = VggHook(vgg)
    tfs = transforms.ToTensor()
    resize = transforms.Resize(size=(50, 60))
    vgg.eval().cpu()
    output_dir = Path("./output/vgg/features/")
    image_path = {
        "小狗": r"./datas/小狗.png",
        "小狗2": r"./datas/小狗2.png",
        "小猫": r"./datas/小猫.jpg",
        "小猫2": r"./datas/小猫2.jpg",
        "飞机": r"./datas/飞机.jpg",
        "飞机2": r"./datas/飞机2.jpg",
    }
    for name in image_path.keys():
        img = Image.open(image_path[name]).convert("RGB")  # 加载图像并将图像转换为RGB3通道
        img = tfs(img)
        img = img[None]  # 增加维度从CHW-->NCHW

        scores = vgg(img)
        pre_index = torch.argmax(scores, dim=1)
        prob = torch.softmax(scores, dim=1)
        top5 = torch.topk(prob, k=5, dim=1)
        print(name)
        print(top5)

        _output_dir = output_dir / name
        _output_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in vgg_hooks.images.keys():
            features = vgg_hooks.images[layer_idx]
            n, c, h, w = features.shape
            for i in range(n):
                imgs = features[i: i+1]  # NCHW --> 1CHW
                imgs = torch.permute(imgs, (1, 0, 2, 3))  # 1CHW --> C1HW
                imgs = resize(imgs)
                torchvision.utils.save_image(
                    imgs,
                    _output_dir / f"{i}_{layer_idx}.png",
                    nrow=8,
                    padding=5
                )
    vgg_hooks.remove()
