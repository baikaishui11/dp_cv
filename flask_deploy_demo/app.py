from flask import Flask, jsonify, request

from model_predictor import ModelPredictor

app = Flask(__name__)
predictor = ModelPredictor()

@app.route("/")
@app.route("/index")
def index():
    return "简易界面"


@app.route("/tt_params", methods=["GET", "POST"])
@app.route("/tt_params/<string:name>/<int:age>", methods=["GET"])
def tt_params(name="xiaoming", age=16):
    if request.method == "GET":
        _args = request.args
    elif request.method == "POST":
        _args = request.form
    else:
        raise ValueError("仅支持POST和GET请求，当前异常实际不会触发")

    print(f"参数类型:{_args}--{type(_args)}")
    adress = _args.get("adress", "默认为上海")
    return jsonify({
        "code": 200,
        "msg": "成功",
        "name": name,
        "age": age,
        "adress": adress
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        _args = request.args
    elif request.method == "POST":
        _args = request.form
    else:
        raise ValueError("仅支持POST和GET请求，当前异常实际不会触发")
    # 根据接口解析参数，例如将base64字符串转化为图片在传入模型进行预测
    _value = _args.get("value")
    if _value is None:
        return jsonify({"code":201, "msg": "请给定有效参数"})

    _result = predictor.predictor(_value)
    return jsonify({
        "code": 202,
        "msg": "成功",
        "data": _result
    })
