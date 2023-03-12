from torch import nn

from cifar10_trainer import CIFAR10Trainer


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


if __name__ == '__main__':
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    trainer = CIFAR10Trainer(model=model)

    trainer.fit(max_epochs=100)
