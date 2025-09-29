import os.path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.w1 = nn.Parameter(torch.empty(8, 6))
        self.b1 = nn.Parameter(torch.empty(6))
        self.w2 = nn.Parameter(torch.empty(6, 3))
        self.b2 = nn.Parameter(torch.empty(3))
        self.w3 = nn.Parameter(torch.empty(3, 1))
        self.b3 = nn.Parameter(torch.empty(1))

        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)
        nn.init.kaiming_uniform_(self.w3)

        nn.init.constant_(self.b1, 0.1)
        nn.init.constant_(self.b2, 0.1)
        nn.init.constant_(self.b3, 0.1)

    def forward(self, x):
        x = x @ self.w1 + self.b1
        x = torch.sigmoid(x)
        x = x @ self.w2 + self.b2
        x = torch.sigmoid(x)
        x = x @ self.w3 + self.b3
        return x


def t1():
    model = Net1()
    x = torch.rand((8, 8))
    r = model(x)
    print(r)


def training():
    california = fetch_california_housing()
    X = california.data
    Y = california.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=66)
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train).astype("float32")
    x_test = x_scaler.transform(x_test).astype("float32")
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype("float32")
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).astype("float32")
    # print(x_train.shape)
    # print(y_train.shape)

    net = Net1()
    loss_fn = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr=0.01)
    total_epoch = 100
    batch_size = 16
    total_batch = len(x_train) // batch_size
    net.train()
    for epoch in range(total_epoch):
        rnd_index = np.random.permutation(len(x_train))
        for batch in range(total_batch):
            _index = rnd_index[batch * batch_size: (batch+1) * batch_size]
            _x = torch.from_numpy(x_train[_index])
            _y = torch.from_numpy(y_train[_index])
            _y_pre = net(_x)
            _loss = loss_fn(_y_pre, _y)

            opt.zero_grad()
            _loss.backward()
            opt.step()
            print(f"epoch:{epoch}, batch:{batch}, loss:{_loss:.4f}")

    net.eval()
    with torch.no_grad():
        y_test_pre = net(torch.from_numpy(x_test))
        test_loss = loss_fn(y_test_pre, torch.from_numpy(y_test))
        print(y_test_pre.detach().numpy())
        print(test_loss.item())

    path = "./output/net.pkl"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({'net': net,
                'total_epoch': total_epoch,
                'lr': 0.01,
                'opt': opt}, path)
    obj = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    print(obj)


if __name__ == "__main__":
    training()
    # t1()

