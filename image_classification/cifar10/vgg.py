import torch
from torch import nn

from cifar10_trainer import CIFAR10Trainer

if __name__ == '__main__':
    model = nn.Sequential(
        nn.LazyConv2d(16, kernel_size=3, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.LazyConv2d(32, kernel_size=3, padding=1, bias=False),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.LazyConv2d(32, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.LazyConv2d(32, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.LazyConv2d(128, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.LazyConv2d(128, kernel_size=3, padding=1, bias=False),
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=(2, 2)),

        nn.Flatten(),
        nn.LazyLinear(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.LazyLinear(10),
    )

    model(torch.randn(1, 3, 32, 32))

    print(model)

    trainer = CIFAR10Trainer(model=model)
    trainer.fit(max_epochs=250)
