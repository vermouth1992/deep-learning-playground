import torch
from torch import nn

from tqdm.auto import trange


def train_nets(iterations=100):
    x = torch.randn(100, 10)
    y = torch.randn(100, 2)
    net = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    for _ in trange(iterations):
        optimizer.zero_grad()
        out = net(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    num_threads = 2
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    train_nets(100)

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    train_nets(100)
