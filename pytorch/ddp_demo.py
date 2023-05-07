import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


import pytorch_lightning as pl

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        output = self.net2(self.relu(self.net1(x)))
        torch.distributed.all_reduce(output)
        return output

import transformers

transformers.BloomModel

def demo_basic(rank, world_size):
    setup(rank, world_size)
    print(f"Running basic DDP example on rank {torch.distributed.get_rank()}.")

    # create model and move it to GPU with id rank
    model = ToyModel()
    # model = torch.compile(model)
    ddp_model = DDP(model, device_ids=None)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5)
    loss = loss_fn(outputs, labels)
    print(f'Rank {torch.distributed.get_rank()}, loss {loss}')
    loss.backward()
    print(f'Rank {torch.distributed.get_rank()}, loss {loss}')
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    run_demo(demo_basic, world_size=2)
