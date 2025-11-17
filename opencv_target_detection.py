import pickle
from glob import glob

import numpy as np
import cv2 as cv
from pathlib import Path
from tqdm import *
from skimage.segmentation import felzenszwalb


def stage1():
    out_dir = Path("./output/01/dog/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv.imread("./datas/dog.png")
    h, w, _ = img.shape
    scales = [
        [400, 450]
    ]
    rk = 10
    for kh, kw in tqdm(scales):
        for sh in range(0, h - kh, rk):
            eh = sh + kh
            for sw in range(0, w - kw, rk):
                ew = sw + kw

                cv.imwrite(str(out_dir / f"{kh}_{kw}_{sh}_{sw}.png"), img[sh:eh, sw:ew, :])


def stage1_felzenszwalb():
    out_dir = Path("./output/01/dog/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv.imread("./datas/dog.png")

    img_mask = felzenszwalb(img)
    cv.imshow("img", img)
    cv.imshow("img_mask", img_mask / img_mask.max())

    cv.waitKey(0)
    cv.destroyAllWindows()



def stage2():
    def _img_feature(_img):
        return np.bincount(_img.ravel(), minlength=256)

    features = []
    img_files = glob("./output/01/dog/images/*.png")
    out_dir = Path("./output/01/dog")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_file in tqdm(img_files):
        img = cv.imread(img_file)
        features.append(_img_feature(img))

    features = np.asarray(features)
    with open(str(out_dir / "features.pkl"), "wb") as writer:
        pickle.dump({"files": img_files, "features": features}, writer)


def stage3():
    out_dir = Path("./output/01/dog")
    with open(str(out_dir / "features.pkl"), "rb") as reader:
        obj = pickle.load(reader)
        img_files = obj["files"]
        features = obj["features"]


if __name__ == '__main__':
    # stage1()
    # stage2()
    stage1_felzenszwalb()