import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, out_dim)

    @torch.compile
    def function(self, x, weight1, bias1, weight2, bias2, weight3, bias3):
        x = torch.nn.functional.linear(x, weight1, bias1)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight2, bias2)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight3, bias3)
        return x

    def forward(self, x):
        return self.function(x, self.layer1.weight, self.layer1.bias, self.layer2.weight, self.layer2.bias,
                             self.layer3.weight, self.layer3.bias)

        # x = self.layer1(x)
        # x = torch.nn.functional.relu(x)
        # x = self.layer2(x)
        # x = torch.nn.functional.relu(x)
        # x = self.layer3(x)
        # return x


import deepspeed

config = {
  "train_batch_size": 16,
  "steps_per_print": 2000,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": False,
  "fp16": {
      "enabled": True,
      "fp16_master_weights_and_grads": False,
      "loss_scale": 0,
      "loss_scale_window": 500,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 15
  },
  "wall_clock_breakdown": False,
  "zero_optimization": {
      "stage": 3,
      "allgather_partitions": True,
      "reduce_scatter": True,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "cpu_offload": True
  }
}

from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, total_length, input_size, total_labels) -> None:
        super().__init__()
        self.total_length = total_length
        self.input_size = input_size
        self.total_labels = total_labels

    def __getitem__(self, index):
        image =  torch.randn(size=(self.input_size,), dtype=torch.float32)
        label = torch.randint(low=0, high=self.total_labels, size=(), dtype=torch.int64)
        return image, label

    def __len__(self):
        return self.total_length

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    deepspeed.init_distributed()
    net = MLP(in_dim=32, out_dim=10)
    training_data = DummyDataset(total_length=10000, input_size=32, total_labels=10)
    
    # net = torch.compile(net)

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        model=net, model_parameters=net.parameters(), training_data=training_data, config=config)

    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')

    log_interval = 100

    for epoch in range(10):
        running_loss = 0.0
        for i, (image, label) in enumerate(trainloader):
            image = image.to(model_engine.local_rank)
            label = label.to(model_engine.local_rank)
            
            if fp16:
                image = image.half()

            outputs = model_engine(image)
            loss = criterion(outputs, label)

            model_engine.backward(loss)
            model_engine.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == (
                    log_interval -
                    1):  # print every log_interval mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / log_interval))
                running_loss = 0.0

# end main
