import numpy as np
from dp_cv.iris_processor import IrisProcessor

processor = IrisProcessor(r"F:\machine\git_hub\dp_cv\output\01\251023211823\model\best.pt")
while True:
    x = input("请输入特征属性，用空格隔开:")
    if x == "q":
        break
    x = x.split(" ")
    if len(x) != 4:
        print(f"输入特征属性异常, 请输入四维特征属性:{x}")
        continue
    x = np.array([x], dtype=float)
    r = processor.predict(x)
    print(f"预测结果为:{r}")

