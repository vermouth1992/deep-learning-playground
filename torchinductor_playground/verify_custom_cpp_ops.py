import copy

import torch

torch.ops.load_library("custom_cuda_op/build/lib.linux-x86_64-cpython-310/my_sigmoid.cpython-310-x86_64-linux-gnu.so")
print(torch.ops.my_ops.sigmoid_forward)
print(torch.ops.my_ops.sigmoid_backward)


class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.ops.my_ops.sigmoid_forward(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, dout):
        out, = ctx.saved_tensors
        return torch.ops.my_ops.sigmoid_backward(out, dout)

def my_sigmoid(x):
    return MySigmoid.apply(x)


if __name__ == '__main__':
    # verify my_sigmoid
    x = torch.randn(5, requires_grad=True, device='cuda', dtype=torch.float32)
    x_copy = copy.deepcopy(x)

    y = torch.sigmoid(x)
    y.sum().backward()
    #
    my_y = my_sigmoid(x_copy)
    my_y.sum().backward()
    #
    print(y, my_y)
    print(x.grad, x_copy.grad)
