# -*- coding: utf-8 -*-
"""
Neural Networks
===============

Neural networks can be constructed using the ``torch.nn`` package.

Now that you had a glimpse of ``autograd``, ``nn`` depends on
``autograd`` to define models and differentiate them.
An ``nn.Module`` contains layers, and a method ``forward(input)``\ that
returns the ``output``.

For example, look at this network that classfies digit images:

.. figure:: /_static/img/mnist.png
   :alt: convnet

   convnet

It is a simple feed-forward network. It takes the input, feeds it
through several layers one after the other, and then finally gives the
output.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or
  weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule:
  ``weight = weight + learning_rate * gradient``

Define the network
------------------

Let’s define this network:
"""
import progressbar
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as data_utils
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        init.kaiming_normal(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        init.kaiming_normal(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_dropout = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(512, 10)
        init.kaiming_normal(self.fc2.weight)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), (2, 2))
        x = self.dropout1(x)
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x

    def check_accuracy_loss(self, x, y, criterion=None):
        """
        :param x: a Tensor of shape (N, C, H, W) of 
        :param y: a Tensor of shape (N,)
        """
        dataset = data_utils.TensorDataset(data_tensor=x, target_tensor=y)
        loader = data_utils.DataLoader(dataset=dataset, batch_size=min(256, x.size(0)))
        total, correct = 0, 0
        if criterion is not None:
            total_loss = 0.0
        for images, labels in loader:
            images = Variable(images)
            # check accuracy
            scores = net.forward(images)
            _, predicted = torch.max(scores.data, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)
            # check loss
            if criterion is not None:
                loss = criterion(scores, Variable(labels))
                total_loss += loss.data[0]
        if criterion is not None:
            return float(correct) / float(total), float(total_loss) / float(len(loader))
        return float(correct) / float(total)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# train LeNet-5 on cifar-10
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from utils.data_utils import *

    cifar10_data = get_CIFAR10_data()

    for key in cifar10_data:
        cifar10_data[key] = torch.from_numpy(cifar10_data[key])

    X_train = cifar10_data['X_train']
    y_train = cifar10_data['y_train']
    X_val = cifar10_data['X_val']
    y_val = cifar10_data['y_val']
    X_test = cifar10_data['X_test']
    y_test = cifar10_data['y_test']
    X_dev = cifar10_data['X_dev']
    y_dev = cifar10_data['y_dev']

    train_dataset = data_utils.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

    validation_dataset = data_utils.TensorDataset(X_val, y_val)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=256)

    # create your optimizer
    net = Net()
    print "Number of trainable parameters: %d" % (len(list(net.parameters())))

    raw_input("Press any key to proceed...")

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    num_training_samples = X_train.size(0)

    for epoch in range(num_epochs):

        # setup progressbar
        bar = progressbar.ProgressBar(redirect_stdout=False, max_value=num_training_samples)
        current_processed_num = 0

        # in your training loop:
        total_loss = 0
        for images, labels in train_loader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()  # zero the gradient buffers
            scores = net.forward(images)
            loss = criterion(scores, labels)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()  # Does the update

            # progress bar
            current_processed_num += images.size()[0]
            bar.update(current_processed_num)

        bar.finish()

        training_accuracy = net.check_accuracy_loss(X_train, y_train)
        training_loss = total_loss / num_training_samples

        validation_accuracy, validation_loss = net.check_accuracy_loss(X_val, y_val, criterion=criterion)

        print ('Epoch [%d/%d] loss: %.4f, accuracy: %0.4f, validation loss: %0.4f, validation accuracy: %.4f'
               % (epoch + 1, num_epochs, training_loss, training_accuracy, validation_loss, validation_accuracy))

    # test
    total, correct = 0.0, 0.0
    test_accuracy = net.check_accuracy_loss(X_test, y_test)
    print ("Testing accuracy: %.4f" % (test_accuracy))