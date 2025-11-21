class ModelPredictor(object):
    def __init__(self):
        print("模型恢复")

    def predictor(self, values):
        print("基于模型输入，定义好入参，使用恢复模型对values进行预测，返回模型预测结果")
        return {"class": "类别1", "prob": 0.35}
