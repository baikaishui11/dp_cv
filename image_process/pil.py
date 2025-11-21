import base64
import copy
import io

import numpy as np
from PIL import Image


def t1():
    img = Image.open(r"../datas/小狗.png")
    # img = img.convert("L")  # rgb转灰度
    # img.show()
    print(type(img))
    img.save("d.bmp")
    w, h = img.size
    print(img.format, img.size, img.mode)

    img_arr = np.array(img)
    if len(img_arr.shape) == 2:
        img_arr = img_arr[:, :, None]
    print(img_arr.shape)
    print(img_arr[:2, :2, :])

    img_arr1 = copy.deepcopy(img_arr)
    img_arr1 = img_arr1[:100, :100, :3]
    mode = "RGB"
    if img.mode == "L":
        img_arr1 = img_arr1[:, :, 0]
        mode = "L"
    img1 = Image.fromarray(img_arr1, mode)
    img1.save("img1.png")

    for i in range(5):
        img_arr2 = copy.deepcopy(img_arr)
        s = np.random.randint(50, min(w, h) // 2)
        hi = np.random.randint(0, h - s)
        wi = np.random.randint(0, w - s)
        img_arr2 = img_arr2[:, :, :3]
        img_arr2[hi:hi + s, wi:wi + s:, :3] = 0
        if img.mode == "L":
            img_arr2 = img_arr2[:, :, 0]
            mode = "L"
        img2 = Image.fromarray(img_arr2, mode)
        img2.save(f"img2_{i}.png")


def t2():
    img = Image.open(r"../datas/小狗.png")
    img1 = img.resize((300, 50), resample=Image.NEAREST)
    # img1.show()

    # 旋转
    img2 = img.rotate(angle=20, expand=False, fillcolor=(255, 255, 255))
    # img2.show()

    # 翻转
    img3 = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img3.show()

    # 剪切
    _img = Image.open(r"../datas/小猫.jpg")
    _img.show()
    box = (300, 100, 900, 300)
    img4 = _img.crop(box)
    img4.show()


def t3():
    # 文件转字符串
    # with open(r"../datas/小猫.jpg", "rb") as reader:
    #     img_data = reader.read()
    # img_data = base64.b64encode(img_data)
    # # base64字符串转图像
    # img_data = base64.b64decode(img_data)
    # img = Image.open(io.BytesIO(img_data))
    # img.show()

    # 图像数据转base64
    img = Image.open(r"../datas/小狗.png").convert("RGB")
    img_data = img.tobytes()
    img_data = base64.b64encode(img_data)
    mode = img.mode
    size = img.size
    # base64字符串转图像
    img_data = base64.b64decode(img_data)
    img = Image.frombytes(mode, size, img_data)
    img.show()


if __name__ == '__main__':
    # t1()
    # t2()
    t3()
