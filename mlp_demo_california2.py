import os.path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(nn.Linear(8, 6),
                                   nn.Sigmoid(),
                                   nn.Linear(6, 3),
                                   nn.Sigmoid(),
                                   nn.Linear(3, 1)
                                   )

    def forward(self, x):
        return self.model(x)


class CaliforniaDataset(Dataset):
    def __init__(self, x, y):
        super(CaliforniaDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def fetch_dataloader(batch_size):
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

    train_dataset = CaliforniaDataset(x_train, y_train)
    test_dataset = CaliforniaDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)
    return train_dataloader, test_dataloader


def save_model(path, net, train_epoch, test_epoch):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({'net': net,
                'train_epoch': train_epoch,
                'test_epoch': test_epoch,
                'lr': 0.01
                }, path)


def training(restore_path=None):
    total_epoch = 100
    train_dataloader, test_dataloader = fetch_dataloader(16)
    net = Net()
    loss_fn = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr=0.01)

    if (restore_path is not None) and (os.path.exists(restore_path)):
        original_net = torch.load(restore_path, map_location='cpu', weights_only=False)
        net.load_state_dict(state_dict=original_net["net"].state_dict())
        train_batch = original_net["train_batch"]
        test_batch = original_net["test_batch"]
        start_epoch = original_net["epoch"] + 1
        total_epoch = total_epoch + start_epoch
    else:
        train_batch = 0
        test_batch = 0
        start_epoch = 0
    writer = SummaryWriter('./output/summary01')
    writer.add_graph(net, torch.rand(1, 8))

    for epoch in range(start_epoch, total_epoch):
        net.train()
        train_loss = []
        for _x, _y in train_dataloader:
            _y_pre = net(_x)
            _loss = loss_fn(_y_pre, _y)

            opt.zero_grad()
            _loss.backward()
            opt.step()
            train_batch += 1
            train_loss.append(_loss.item())

            print(f"train epoch:{epoch}, batch:{train_batch}, loss:{_loss:.4f}")
            writer.add_scalar("train_batch_loss", _loss.item(), global_step=train_batch)

        net.eval()
        test_loss = []
        with torch.no_grad():
            for _x, _y in test_dataloader:
                _y_pre = net(_x)
                _loss = loss_fn(_y_pre, _y)
                test_batch += 1

                print(f"test epoch:{epoch}, batch:{test_batch}, loss:{_loss:.4f}")
                writer.add_scalar("test_batch_loss", _loss.item(), global_step=test_batch)
                test_loss.append(_loss.item())

        writer.add_histogram("w1", net.model[0].weight, global_step=epoch)
        writer.add_histogram("b1", net.model[0].bias, global_step=epoch)
        writer.add_scalars("loss", {"train": np.mean(train_loss), "test": np.mean(test_loss)}, global_step=epoch)

        if epoch % 100 == 0:
            save_model(f"./output/net{epoch}.pkl", train_batch, test_batch)
    writer.close()


if __name__ == "__main__":
    training()
    # t1()

