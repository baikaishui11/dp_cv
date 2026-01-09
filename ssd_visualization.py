import torch
from torch import Tensor
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2 as cv
import torch.optim as optim


def train(model):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001)

    images: list[Tensor] = [
        torch.rand(3, 1300, 1300),
        torch.rand(3, 365, 387),
        torch.rand(3, 300, 300),
        torch.rand(3, 321, 450)
    ]
    targets: list[dict[str, Tensor]] = []
    boxes = torch.tensor(
        [
            [[8.8069e+01, 1.4674e+02, 5.6282e+02, 3.7051e+02],
             [5.8338e+02, 2.6097e+02, 6.0366e+02, 3.3450e+02],
             [1.6025e-01, 2.7076e+02, 1.7438e+01, 3.6145e+02],
             [5.5874e+02, 2.7589e+02, 5.7230e+02, 3.3554e+02],
             [6.0216e+02, 2.6434e+02, 6.2211e+02, 3.2958e+02],
             [6.2616e+02, 2.6423e+02, 6.3996e+02, 3.2847e+02],
             [1.7602e+02, 1.8451e+02, 1.9516e+02, 2.0253e+02],
             [1.2418e+01, 2.8615e+02, 3.9832e+01, 3.1260e+02],
             [1.4647e+02, 1.8438e+02, 1.6404e+02, 2.0427e+02],
             [1.0756e+01, 2.2248e+02, 2.2998e+01, 2.5238e+02],
             [3.0237e+02, 2.7734e+02, 3.1222e+02, 2.9777e+02],
             [1.2737e+02, 1.9033e+02, 1.4161e+02, 2.0765e+02],
             [2.4141e+02, 1.7469e+02, 2.6035e+02, 1.9441e+02],
             [4.2691e+02, 2.6918e+02, 4.3745e+02, 2.8266e+02]],
            [[8.8069e+01, 1.4674e+02, 5.6282e+02, 3.7051e+02],
             [5.8338e+02, 2.6097e+02, 6.0366e+02, 3.3450e+02],
             [1.6025e-01, 2.7076e+02, 1.7438e+01, 3.6145e+02],
             [5.5874e+02, 2.7589e+02, 5.7230e+02, 3.3554e+02],
             [6.0216e+02, 2.6434e+02, 6.2211e+02, 3.2958e+02],
             [6.2616e+02, 2.6423e+02, 6.3996e+02, 3.2847e+02],
             [1.7602e+02, 1.8451e+02, 1.9516e+02, 2.0253e+02],
             [1.2418e+01, 2.8615e+02, 3.9832e+01, 3.1260e+02],
             [1.4647e+02, 1.8438e+02, 1.6404e+02, 2.0427e+02],
             [1.0756e+01, 2.2248e+02, 2.2998e+01, 2.5238e+02],
             [3.0237e+02, 2.7734e+02, 3.1222e+02, 2.9777e+02],
             [1.2737e+02, 1.9033e+02, 1.4161e+02, 2.0765e+02],
             [2.4141e+02, 1.7469e+02, 2.6035e+02, 1.9441e+02],
             [4.2691e+02, 2.6918e+02, 4.3745e+02, 2.8266e+02]],
            [[8.8069e+01, 1.4674e+02, 5.6282e+02, 3.7051e+02],
             [5.8338e+02, 2.6097e+02, 6.0366e+02, 3.3450e+02],
             [1.6025e-01, 2.7076e+02, 1.7438e+01, 3.6145e+02],
             [5.5874e+02, 2.7589e+02, 5.7230e+02, 3.3554e+02],
             [6.0216e+02, 2.6434e+02, 6.2211e+02, 3.2958e+02],
             [6.2616e+02, 2.6423e+02, 6.3996e+02, 3.2847e+02],
             [1.7602e+02, 1.8451e+02, 1.9516e+02, 2.0253e+02],
             [1.2418e+01, 2.8615e+02, 3.9832e+01, 3.1260e+02],
             [1.4647e+02, 1.8438e+02, 1.6404e+02, 2.0427e+02],
             [1.0756e+01, 2.2248e+02, 2.2998e+01, 2.5238e+02],
             [3.0237e+02, 2.7734e+02, 3.1222e+02, 2.9777e+02],
             [1.2737e+02, 1.9033e+02, 1.4161e+02, 2.0765e+02],
             [2.4141e+02, 1.7469e+02, 2.6035e+02, 1.9441e+02],
             [4.2691e+02, 2.6918e+02, 4.3745e+02, 2.8266e+02]],
            [[8.8069e+01, 1.4674e+02, 5.6282e+02, 3.7051e+02],
             [5.8338e+02, 2.6097e+02, 6.0366e+02, 3.3450e+02],
             [1.6025e-01, 2.7076e+02, 1.7438e+01, 3.6145e+02],
             [5.5874e+02, 2.7589e+02, 5.7230e+02, 3.3554e+02],
             [6.0216e+02, 2.6434e+02, 6.2211e+02, 3.2958e+02],
             [6.2616e+02, 2.6423e+02, 6.3996e+02, 3.2847e+02],
             [1.7602e+02, 1.8451e+02, 1.9516e+02, 2.0253e+02],
             [1.2418e+01, 2.8615e+02, 3.9832e+01, 3.1260e+02],
             [1.4647e+02, 1.8438e+02, 1.6404e+02, 2.0427e+02],
             [1.0756e+01, 2.2248e+02, 2.2998e+01, 2.5238e+02],
             [3.0237e+02, 2.7734e+02, 3.1222e+02, 2.9777e+02],
             [1.2737e+02, 1.9033e+02, 1.4161e+02, 2.0765e+02],
             [2.4141e+02, 1.7469e+02, 2.6035e+02, 1.9441e+02],
             [4.2691e+02, 2.6918e+02, 4.3745e+02, 2.8266e+02]]
        ],
    )
    labels = torch.randint(1, 91, (4, 14))
    for idx in range(len(images)):
        image_y = {
            "boxes": boxes[idx],
            "labels": labels[idx]
        }
        targets.append(image_y)
    loss = model(images, targets)
    print(loss)
    loss = loss['classification'] + 0.58 * loss['bbox_regression']
    opt.zero_grad()
    loss.backward()
    opt.step()



def inference(model):
    model.eval()

    # img0 = Image.open("../images/a.jpeg")
    img0 = Image.open("./images/c.jpg")
    ts = transforms.Compose([
        transforms.ToTensor()
    ])
    img = ts(img0)[None, ...]
    r = model(img)
    print(r)
    r = r[0]

    img0 = cv.cvtColor(np.asarray(img0).astype(np.uint8), cv.COLOR_RGB2BGR)
    boxes = r['boxes'].detach().numpy()
    labels = r['labels'].detach().numpy()
    scores = r['scores'].detach().numpy()
    print(f"总预测边框数目:{len(labels)}")
    label_2_color = {}
    for label in np.unique(labels):
        try:
            color = label_2_color[label]  # 不同类别的边框使用不同的颜色
        except KeyError:
            color = list(map(int, np.random.randint(255, size=(3,))))
            label_2_color[label] = color
        idx = labels == label
        for box, score in zip(boxes[idx], scores[idx]):
            box = list(map(int, box))
            # 绘制边框 + 文字描述
            cv.rectangle(img0, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=color, thickness=2)
            cv.putText(img0, text=f"{label}:{score:.3f}", org=(box[0], box[1]), color=(255, 255, 255), thickness=2,
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, lineType=cv.LINE_AA)

    cv.imwrite("./images/1.png", img0)
    cv.imshow('x', img0)
    cv.waitKey(0)
    cv.destroyAllWindows()


def tt_ssd():
    # coco数据集
    model = models.detection.ssd300_vgg16(
        weights=models.detection.ssd.SSD300_VGG16_Weights.COCO_V1,
        score_thresh=0.1,  # 分类置信度阈值
        nms_thresh=0.45,  # NMS的IoU阈值
        detections_per_img=100,  # 每个图像最终保留最多多少个边框
        topk_candidates=200  # 计算过程中top k的数量
    )
    # print(model)
    # inference(model)
    train(model)


if __name__ == '__main__':
    tt_ssd()

