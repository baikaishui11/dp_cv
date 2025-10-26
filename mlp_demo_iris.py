from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dp_cv.modules.numpy_dateset import build_dataloader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from dp_cv.modules import metrics
import onnxruntime as ort


class IrisNetwork(nn.Module):
    def __init__(self):
        super(IrisNetwork, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.classify(x)


def save(obj, path):
    torch.save(obj, path)


def load(path, net):
    print(f"模型恢复{path}")
    ss_mode = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(state_dict=ss_mode["net"].state_dict(), strict=True)
    start_epoch = ss_mode["epoch"] + 1
    train_batch = ss_mode["train_batch"]
    test_batch = ss_mode["test_batch"]
    best_acc = ss_mode["acc"]
    return start_epoch, train_batch, test_batch, best_acc


def training():
    # now = datetime.now().strftime("%y%m%d%H%M%S")
    now = 251023211823
    root_dir = Path(f"./output/01/{now}")
    summary_dir = root_dir / "summary"
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    checkout_dir = root_dir / "model"
    if not checkout_dir.exists():
        checkout_dir.mkdir(parents=True)
    last_path = checkout_dir / "last.pkl"
    best_path = checkout_dir / "best.pkl"

    total_epoch = 100
    start_epoch = 0
    summary_interval_batch = 2
    save_interval_batch = 2
    train_batch = 0
    test_batch = 0
    best_acc = -1.0

    iris = load_iris()
    X = iris.data.astype("float32")
    Y = iris.target.astype("int64")
    train_dataloader, test_dataloader, x_test, y_test = build_dataloader(X, Y, test_size=0.1, batch_size=32)

    net = IrisNetwork()
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = metrics.Accuracy()
    opt = optim.SGD(net.parameters(), lr=0.005)

    if best_path.exists():
        start_epoch, train_batch, test_batch, best_acc = load(best_path, net)
    elif last_path.exists():
        start_epoch, train_batch, test_batch, best_acc = load(last_path, net)

    writer = SummaryWriter(summary_dir)
    writer.add_graph(net, torch.rand(3, 4))

    for epoch in range(start_epoch, total_epoch+start_epoch):
        net.train()
        train_losses = []
        train_true, train_total = 0, 0
        for x, y in train_dataloader:
            score = net(x)
            loss = loss_fn(score, y)
            n, acc = acc_fn(score, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += n
            train_true += n * acc

            if train_batch % summary_interval_batch == 0:
                print(f"epoch:{epoch}, train batch:{train_batch}, loss:{loss.item():.3f}, acc:{acc.item():.3f}")
                writer.add_scalar("train_loss", loss.item(), global_step=train_batch)
                writer.add_scalar("train_acc", acc.item(), global_step=train_batch)
            train_batch += 1
            train_losses.append(loss.item())

        net.eval()
        test_losses = []
        test_true, test_total = 0, 0
        with torch.no_grad():
            for x, y in train_dataloader:
                socres = net(x)
                loss = loss_fn(socres, y)
                n, acc = acc_fn(socres, y)

                test_total += n
                test_true += n * acc

                print(f"epoch:{epoch}, test batch:{test_batch}, loss:{loss.item():.3f}, acc:{acc.item():.3f}")
                writer.add_scalar("test_loss", loss.item(), global_step=test_batch)
                writer.add_scalar("test_acc", acc.item(), global_step=test_batch)
                test_batch += 1
                test_losses.append(loss.item())

        writer.add_scalars("loss", {"train": np.mean(train_losses), "test": np.mean(test_losses)}, global_step=epoch)
        writer.add_scalars("acc", {"train": train_true / train_total, "test": test_true / test_total}, global_step=epoch)

        if (test_true / test_total) > best_acc:
            obj = {
                "net": net,
                "epoch": epoch,
                "train_batch": train_batch,
                "test_batch": test_batch,
                "acc": test_true / test_total
            }
            save(obj, best_path.absolute())
            best_acc = test_true / test_total
        if epoch % save_interval_batch == 0:
            obj = {
                "net": net,
                "epoch": epoch,
                "train_batch": train_batch,
                "test_batch": test_batch,
                "acc": test_true / test_total
            }
            save(obj, last_path.absolute())
    obj = {
        "net": net,
        "epoch": start_epoch + total_epoch - 1,
        "train_batch": train_batch,
        "test_batch": test_batch,
        "acc": test_true / test_total
    }
    save(obj, last_path.absolute())
    writer.close()


def tt_model():
    best = torch.load(r"./output/01/251023211823/model/best.pkl", map_location="cpu", weights_only=False)
    last = torch.load(r"./output/01/251023211823/model/last.pkl", map_location="cpu", weights_only=False)
    print(best["epoch"], best["acc"])
    print(last["epoch"], last["acc"])


def export(model_dir):
    model_dir = Path(model_dir)

    net = torch.load(model_dir / "best.pkl", map_location="cpu", weights_only=False)["net"]
    net.eval().cpu()

    example = torch.rand(1, 4)
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(model_dir / "best.pt")

    # onnx_model = torch.onnx.export(
    #     model=net,
    #     args=example,
    #     dynamo=True,
    # )
    # onnx_model.save(model_dir / "best.onnx")
    torch.onnx.export(
        model=net,
        args=example,
        f=model_dir / "best_dynamic.onnx",
        input_names=["features"],
        output_names=["label"],
        dynamic_axes={
            "features": {
                0: "batch"
            },
            "label": {
                0: "batch"
            }
        }
    )


@torch.no_grad()
def tt_load_model(model_dir):
    model_dir = Path(model_dir)

    net1 = torch.load(model_dir / "best.pkl", map_location="cpu", weights_only=False)["net"]
    net1.eval().cpu()

    example = torch.rand(2, 4)
    net2 = torch.jit.load(model_dir / "best.pt", map_location="cpu")
    net2.eval().cpu()

    # ort_session = ort.InferenceSession(str(model_dir / "best.onnx"))
    # ort_input = {ort_session.get_inputs()[0].name: example.detach().numpy()}
    # ort_outputs = ort_session.run(None, ort_input)
    net3_session = ort.InferenceSession(str(model_dir / "best_dynamic.onnx"))

    print(net1(example))
    print(net2(example))
    # print(ort_outputs[0])
    print(net3_session.run(["label"], input_feed={"features": example.detach().numpy()}))


if __name__ == '__main__':
    # training()
    # tt_model()
    # export(r"./output/01/251023211823/model")
    tt_load_model(r"./output/01/251023211823/model")
