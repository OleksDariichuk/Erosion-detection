import numpy as np
import pandas as pd
import torch

from train import model, device, train_data


def test_run(j):
    for i in range(j):
        data = train_data.sample(1)
        x = torch.from_numpy(np.array(data.values[0][:-1], dtype=np.float32))
        x = x.to(device)
        out, _ = model(x, None)
        print(f"Prediction: {out.item()} | Actual: {data.values[0][-1]}\n")


def write_res(filename='submission.csv'):
    res = []
    for i in range(len(test_data)):
        x = torch.from_numpy(np.array(test_data.values[i], dtype=np.float32))
        x = x.to(device)
        out, _ = model(x, None)
        res.append(out.item())

    pd.DataFrame(res).to_csv(filename, index=False, header=False)


if __name__ == '__main__':
    model.load_state_dict(torch.load('model_2023-03-10T15-52.pth'))
    test_data = pd.read_csv('internship_hidden_test.csv')
    test_run(10)
    write_res()
