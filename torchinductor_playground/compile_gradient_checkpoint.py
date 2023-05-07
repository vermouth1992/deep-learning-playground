import torch
import torch.utils.checkpoint
import torch._dynamo
import argparse
import torch.utils.benchmark as benchmark


class myModel(torch.nn.Module):
    def __init__(self, grad_checkpoint):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.checkpoint = grad_checkpoint

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.utils.checkpoint.checkpoint(self.conv2, x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        return x


def run_forward(model_, x):
    out = model_(x)


def run(grad_checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = myModel(grad_checkpoint).to(device)
    x = torch.randn((2, 3, 640, 256), device=device)

    model_opt = torch.compile(model, mode="reduce-overhead")
    num_threads = torch.get_num_threads()
    t = benchmark.Timer(
        stmt='optim(x)',
        globals={'optim': model_opt, 'x': x}, # When 'optim': model then it works
        num_threads=num_threads,
        label="Average Run Duration",
    )
    print(t.timeit(100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_checkpoint", action='store_true')
    args = parser.parse_args()
    run(args.grad_checkpoint)