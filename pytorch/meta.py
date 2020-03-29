"""
In meta learning, a module should be functional, such that it can perform computation given parameters and inputs.
"""

import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Task(ABC):
    """
    A task contains a train dataset and a test dataset. Usually, it is simply several examples.
    """

    @abstractmethod
    def training_set(self):
        raise NotImplemented

    @abstractmethod
    def testing_set(self):
        raise NotImplemented


class SineWaveTask(Task):
    def __init__(self, k, N):
        """ k-shot, N-way regression """
        self.a = np.random.uniform(0.1, 5.0)
        self.b = np.random.uniform(0, 2 * np.pi)
        self.f = lambda x: self.a * np.sin(x + self.b)
        self.training_x = np.random.uniform(-5, 5, (k, 1))
        self.training_y = self.f(self.training_x)
        self.testing_x = np.linspace(-5, 5, N).reshape((N, 1))
        self.testing_y = self.f(self.testing_x)

        self.training_x = torch.as_tensor(self.training_x, dtype=torch.float32)
        self.training_y = torch.as_tensor(self.training_y, dtype=torch.float32)
        self.testing_x = torch.as_tensor(self.testing_x, dtype=torch.float32)
        self.testing_y = torch.as_tensor(self.testing_y, dtype=torch.float32)

    def training_set(self):
        return self.training_x, self.training_y

    def testing_set(self):
        return self.testing_x, self.testing_y


class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_params(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_params(rest, param)
                    break
        else:
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = param.data.clone().detach().requires_grad_(True)
            self.set_params(name, param)


class GradLinear(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super(GradLinear, self).__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.weights = ignore.weight.data.clone().detach().requires_grad_(True)
        self.bias = ignore.bias.data.clone().detach().requires_grad_(True)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


class SineModel(ModifiableModule):
    def __init__(self):
        super(SineModel, self).__init__()
        self.hidden1 = GradLinear(1, 40)
        self.hidden2 = GradLinear(40, 40)
        self.out = GradLinear(40, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)

    def named_submodules(self):
        return [('hidden1', self.hidden1), ('hidden2', self.hidden2), ('out', self.out)]


def maml(model_cls, training_tasks, loss_fn, lr_outer=1e-3, lr_inner=0.01, num_epochs=4, batch_size=10):
    meta_model = model_cls()
    meta_optimizer = torch.optim.Adam(meta_model.params(), lr=lr_outer)
    meta_optimizer.zero_grad()
    loss = []
    for epoch in range(num_epochs):
        random.shuffle(training_tasks)
        for i, task in enumerate(tqdm(training_tasks, desc='Epoch {}/{}'.format(epoch + 1, num_epochs))):
            train_x, train_y = task.training_set()
            test_x, test_y = task.testing_set()
            # copy model, but on the same parameter reference
            new_model = model_cls()
            new_model.copy(meta_model, same_var=True)
            # update model using meta training set
            meta_train_loss = loss_fn(new_model.forward(train_x), train_y)
            meta_train_loss.backward(retain_graph=True, create_graph=True)
            for name, param in new_model.named_params():
                grad = param.grad
                new_model.set_params(name, param - lr_inner * grad)
            # compute the loss on meta testing set
            meta_test_loss = loss_fn(new_model.forward(test_x), test_y)
            meta_test_loss.backward(retain_graph=True)
            loss.append(meta_test_loss.item())
            if (i + 1) % batch_size == 0:
                meta_optimizer.step()
                meta_optimizer.zero_grad()
    return meta_model, loss


def reptile(model_cls, training_tasks, loss_fn, lr_outer=1e-3, lr_inner=0.01, num_epochs=4, batch_size=10, k=32):
    meta_model = model_cls()
    meta_optimizer = torch.optim.Adam(meta_model.params(), lr=lr_outer)
    meta_optimizer.zero_grad()
    name_to_param = dict(meta_model.named_params())
    loss = []
    for epoch in range(num_epochs):
        random.shuffle(training_tasks)
        for i, task in enumerate(tqdm(training_tasks, desc='Epoch {}/{}'.format(epoch + 1, num_epochs))):
            train_x, train_y = task.training_set()
            # test_x, test_y = task.testing_set()
            # copy model, but NOT on the same parameter reference
            new_model = model_cls()
            new_model.copy(meta_model, same_var=False)
            inner_optimizer = torch.optim.SGD(new_model.params(), lr=lr_inner)
            for _ in range(k):
                inner_optimizer.zero_grad()
                meta_train_loss = loss_fn(new_model.forward(train_x), train_y)
                meta_train_loss.backward()
                inner_optimizer.step()
            # manually compute the gradient w.r.t the meta model parameters
            for name, param in new_model.named_params():
                cur_grad = (name_to_param[name].data - param.data) / k / lr_inner
                if name_to_param[name].grad is None:
                    name_to_param[name].grad = torch.zeros_like(cur_grad)
                name_to_param[name].grad.data.add_(cur_grad / batch_size)
            if (i + 1) % batch_size == 0:
                meta_optimizer.step()
                meta_optimizer.zero_grad()
    return meta_model, loss


if __name__ == '__main__':
    pass
