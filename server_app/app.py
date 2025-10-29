from flask import Flask, request, jsonify
import numpy as np
from iris_processor import IrisProcessor

processor = IrisProcessor(r"./best.pt")
app = Flask(__name__)


@app.route('/')
def index():
    return "Iris数据分类模型接口服务"


@app.route('/predict')
def predict():
    try:
        # get方法情况请求，必须给定参数features，使用','进行特征分割，使用';'进行样本分割
        features = request.args.get("features")
        if features is None:
            return jsonify({"code": 1, "msg": "参数异常， 必须给定有效的features参数"})
        x = [xx.split(",") for xx in features.split(";")]
        x = np.array(x, dtype=float)
        if len(x) == 0:
            return jsonify({"code": 2, "msg": f"参数异常， 必须给定有效的features参数:{features}"})
        if len(x[0]) != 4:
            return jsonify({"code": 2, "msg": f"参数维度异常， 必须给定有效的features参数:{features}"})
        r = processor.predict(x)
        return jsonify({"code": 0, "data": r, "msg": "成功！"})
    except Exception as e:
        return jsonify({"code": 4, "msg": f"服务器异常:{e}"})
