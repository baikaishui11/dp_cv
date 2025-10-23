import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        super(NumpyDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def build_dataloader(X, Y, test_size, batch_size):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=24)

    train_dataset = NumpyDataset(x_train, y_train)
    test_dataset = NumpyDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)
    return train_dataloader, test_dataloader, x_test, y_test
