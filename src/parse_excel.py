import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class StockOptionsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class StockOptionsModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, output_size=1):
        super(StockOptionsModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        # hn: (num_layers, batch, hidden_size)
        hn = hn[-1]  # Take the output from the last LSTM layer
        x = self.fc1(hn)
        # x = self.relu(x)

        return x
    

def import_stock(ticker, days = 10):

    src = pd.read_excel(f"./training_data/{ticker}.xlsx")
    data = src.to_numpy()
    snipped = {}
    count = 0
    for c in src.columns:
        snipped[c] = []
        for i in range(25, data.shape[0]-25):
            snipped[c].append(src[c][i])
        count += 0

    df = pd.DataFrame(snipped)

    X_restruct = []
    y_restruct = []
    traits = []
    features = ["Month Normalized Price", "1-Week Volatility", "2-Week Volatility",
                "Moving Average 1-Week", "Moving Average 2-Week", "Moving Average 3-Week", "Moving Average 4-Week",
                "NASDAQ Month Normalized Price", "NASDAQ 1-Week Volatility", "NASDAQ 2-Week Volatility",
                "NASDAQ Moving Average 1-Week", "NASDAQ Moving Average 2-Week", "NASDAQ Moving Average 3-Week", "NASDAQ Moving Average 4-Week"]
    labels = ["1 Week Change", "2 Week Change", "3 Week Change", "4 Week Change"]
    traits.extend(features)
    traits.extend(labels)

    data = df.to_numpy()

    for i in range(data.shape[0]-days):
        x_features = []
        y_predictions = []
        for t in traits:
            list1 = []
            for j in range(i,i+days):
                list1.append(df[t][j])
            if t in features:
                x_features.append(list1)
            elif t in labels:
                y_predictions.append(df[t][j+1])
        X_restruct.append(x_features)
        y_restruct.append(y_predictions)

    X_restruct = np.array(X_restruct)
    y_restruct = np.array(y_restruct)

    print(X_restruct.shape)
    print(y_restruct.shape)

    return X_restruct, y_restruct


def dataPrep(x, y, train_percentage, batch_size = 32):

    train_test_clipping = int(x.shape[0] * (train_percentage))
    X_train = x[:train_test_clipping]
    X_test = x[train_test_clipping:]
    y_train = y[:train_test_clipping]
    y_test = y[train_test_clipping:]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    train_loader = DataLoader(StockOptionsDataset(X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader(StockOptionsDataset(X_test, y_test), batch_size=len(y_test))

    return train_loader, test_loader, X_train, y_train, X_test, y_test


def train(ticker, days, train_percentage, hidden_size=5, epochs = 20, batch_size = 32, lr = 1e-4):

    x, y = import_stock(ticker, days = days)
    train_loader, test_loader, X_train, y_train, X_test, y_test = dataPrep(x, y, train_percentage, batch_size=batch_size)

    #%% instantiate model, optimizer, and loss
    model = StockOptionsModel(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=y_train.shape[1])
    # input = torch.rand((2, 10, 1))  # BS, seq_len, input_size
    # model(input).shape  # out: [BS, seq_len, hidden]

    #%% Loss and Optimizer
    # loss_fun = nn.MSELoss()
    loss_fun = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    min_loss = 1

    #%% Train
    for epoch in range(epochs):
        for j, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X.view(-1, days, X_train.shape[1]))
            loss = loss_fun(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.data}")
        losses.append(loss.data)
        if epoch%10 == 0 and loss.data < min_loss:
            min_loss = loss.data
            torch.save(model.state_dict(), "./model/VT.pth")
            print(f"Model saved with minimum loss: {loss.data}")

    plt.scatter(np.arange(len(losses)), losses, alpha=0.7)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("./loss_curve.jpg")
    plt.clf()

    model = StockOptionsModel(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=y_train.shape[1])
    model.load_state_dict(torch.load("./model/VT.pth"))
    model.eval()
    test(model, days, test_loader, X_test, y_test)
    
    return model


def test(model, days, test_loader, X_test, y_test):

    test_set = StockOptionsDataset(X_test, y_test)
    X_test_torch, y_test_torch = next(iter(test_loader))
    with torch.no_grad():
        print(X_test_torch.shape)
        y_pred = []
        for i in range(X_test_torch.shape[0]):
            x_seq = X_test_torch[i].view(-1, days, X_test.shape[1])  # shape: [1, 30, 1]
            y_out = model(x_seq)
            y_pred.append(y_out)
        # y_pred = model(torch.unsqueeze(X_test_torch, 2)).detach().squeeze().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().numpy()
    y_act = y_test_torch.squeeze().numpy()
    horizons = ["1 Week", "2 Weeks", "3 Weeks", "4 Weeks"]
    num_horizons = len(horizons)
    x_axis = range(y_act.shape[0])

    y_avg = np.zeros(y_pred.shape) + np.mean(y_act, axis=0)

    plt.figure(figsize=(16, 10))

    for i in range(num_horizons):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_axis, y_act[:, i], label="Actual", alpha=0.7)
        plt.scatter(x_axis, y_pred[:, i], label="Predicted", alpha=0.7, c=np.zeros((y_act.shape[0], 3)) + np.array([0, 1, 0]))
        plt.scatter(x_axis, y_avg[:, i], label="Baseline", alpha=0.7, c=np.zeros((y_act.shape[0], 3)))
        plt.title(f"{horizons[i]} Prediction")
        plt.xlabel("Sample")
        plt.ylabel("Percent Change")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("./performance.jpg")
    plt.close()

    diff = y_act[:] - y_pred[:]
    errors = np.linalg.norm(diff, axis = 1)
    print(np.mean(errors))
    print(np.mean(np.linalg.norm(y_act[:] - y_avg[:], axis = 1)))


train("VT", 30, 0.85, hidden_size=128, epochs = 1000, batch_size = 32, lr=1e-3)