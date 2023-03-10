import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

# set parameters
batch_size = 32
block_size = 53
max_iters = 1_200
learning_rate = 1e-3
eval_interval = 400
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
train = pd.read_csv('internship_train.csv')
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)


def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    xi = torch.randint(len(data), (batch_size,))
    x = torch.stack([torch.from_numpy(np.array(data.values[i][:-1], dtype=np.float32)) for i in xi])
    y = torch.stack([torch.from_numpy(np.array(data.values[i][-1], dtype=np.float32)) for i in xi])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


criterion = RMSELoss()


class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, target):
        out = self.linear(x)
        if target is not None:
            loss = criterion(out, target)
            return out, loss
        else:
            loss = None
        return out, loss


model = RegressionModel(block_size, 1)
model = model.to(device)

# PyTorch optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def get_time():
    return time.strftime("%Y-%m-%dT%H-%M")


if __name__ == '__main__':
    for iter in range(max_iters):
        print(f'iter {iter}')
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'iter {iter} train loss {losses["train"]:.3f} val loss {losses["val"]:.3f}')

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'model_{get_time()}.pth')
