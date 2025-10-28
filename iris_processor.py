"""
iris预测模型处理器
"""
import torch
import torch.nn as nn
import onnxruntime as ort
import os
import numpy as np


def softmax(scores):
    a = np.exp(scores)
    b = np.sum(a, axis=1, keepdims=True)
    p = a / b
    return p


class IrisProcessor(object):
    def __init__(self, model_path):
        super(IrisProcessor, self).__init__()
        model_path = os.path.abspath(model_path)
        _, ext = os.path.splitext(model_path.lower())
        self.pt, self.onnx = False, False
        if ext == ".pt":
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval().cpu()
            self.model = model
            self.pt = True
        elif ext == ".onnx":
            session = ort.InferenceSession(model_path)
            self.session = session
            self.onnx = True
            self.input_name = "features"
            self.output_name = "label"
        else:
            raise ValueError(f"当前只支持pt和onnx格式， 当前文件类型为{model_path}")
        self.classes = ["类别1", "类别2", "类别3"]
        print(f"模型恢复成功：pt-->{self.pt};onnx-->{self.onnx}")

    def _process_after_model(self, x, score):
        """
        后处理逻辑
        :param x: 原始特征属性x，numpy类型[N,4]
        :param score:模型预测的置信度, numpy类型[N,3]
        :return:每个样本返回对应的预测类别名称、id以及概率值，以dict格式返回
        """
        pred_indexes = np.argmax(score, axis=1)
        pred_probas = softmax(score)
        result = []
        for k, idx in enumerate(pred_indexes):
            r = {
                "id": int(idx),
                "label": self.classes[idx],
                "proba": float(pred_probas[k][idx])
            }
            result.append(r)
        return result

    @torch.no_grad()
    def _predict_with_pt(self, x):
        tensor_x = torch.from_numpy(x).to(torch.float)
        scores = self.model(tensor_x)
        scores = scores.numpy()
        return self._process_after_model(x, scores)

    def _predict_with_onnx(self, x):
        onnx_x = x.astype("float32")
        scores = self.session.run([self.output_name], {self.input_name: onnx_x})
        scores = scores[0]
        return self._process_after_model(x, scores)

    def predict(self, x):
        if self.pt:
            return self._predict_with_pt(x)
        elif self.onnx:
            return self._predict_with_onnx(x)
        else:
            raise ValueError("初始化异常")


if __name__ == '__main__':
    processor = IrisProcessor(r"F:\machine\git_hub\dp_cv\output\01\251023211823\model\best.pt")
    r = processor.predict(np.array([[5, 2.3, 1.5, 2.2], [0.2, 1.3, 0.5, 0.2]]))
    print(r)
