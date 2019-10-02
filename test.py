"""
Test to train multivariate Gaussian. We would like to convert a multivariate Guassian to another
multivariate Gaussian
"""

import numpy as np
import torch
import torch.nn as nn
from torchlib.common import FloatTensor
from torchlib.dataset.utils import create_data_loader
from torchlib.trainer.regressor import Regressor


def generate_training_data(x_mean, x_std, y_mean, y_std, num_samples=10000):
    x = np.random.randn(num_samples, len(x_mean)) * np.array(x_std) + np.array(x_mean)
    y = np.random.randn(num_samples, len(y_mean)) * np.array(y_std) + np.array(y_mean)
    x_train = x[:8000]
    y_train = y[:8000]
    x_val = x[8000:]
    y_val = y[8000:]
    return x_train, y_train, x_val, y_val


class Policy(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_feature, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_feature)
        )

        random_number = np.random.randn(out_feature)

        self.logstd = torch.nn.Parameter(
            torch.tensor(random_number, requires_grad=True).type(FloatTensor))

    def forward(self, input):
        batch_size = input.shape[0]
        mean = self.model.forward(input)
        dis = torch.distributions.Normal(mean, torch.exp(self.logstd))
        return dis.rsample(torch.Size([batch_size]))


if __name__ == '__main__':
    x_mean = [0., 1.5]
    x_std = [1, 0]
    y_mean = [-1.5, -0.2]
    y_std = [0.5, 0.1]
    x_train, y_train, x_val, y_val = generate_training_data(x_mean, x_std, y_mean, y_std)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    model = Policy(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    regressor = Regressor(model, optimizer, criterion, scheduler=None)

    train_loader = create_data_loader((x_train, y_train))
    val_loader = create_data_loader((x_val, y_val))

    regressor.train(epoch=100, train_data_loader=train_loader, val_data_loader=val_loader,
                    checkpoint_path=None)
