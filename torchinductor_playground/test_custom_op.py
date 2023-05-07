import copy

import torch

print(torch.__version__)
import torch._dynamo
import torch.fx

import torch._dynamo.backends.inductor

import torch._inductor.config

torch._inductor.config.trace.enabled = True

from verify_custom_cpp_ops import my_sigmoid


def print_backend(gm: torch.fx.GraphModule, input):
    gm.graph.print_tabular()
    return gm


def pattern(x):
    return torch.sigmoid(x)


# def replacement(x):
#     return my_sigmoid(x)


replacement = torch.fx.symbolic_trace(my_sigmoid)
replacement.graph.print_tabular()

def replace_pattern_backend(gm: torch.fx.GraphModule, input):
    gm.graph.print_tabular()
    torch.fx.replace_pattern(gm, pattern=pattern, replacement=replacement)
    gm.graph.print_tabular()
    return gm


def replace_pattern_backend_with_inductor(gm: torch.fx.GraphModule, input):
    torch.fx.replace_pattern(gm, pattern=pattern, replacement=replacement)
    gm.graph.lint()
    gm.recompile()
    from torch._inductor.compile_fx import compile_fx
    gm.graph.print_tabular()
    optimized_forward = compile_fx(gm, example_inputs_=input)
    return optimized_forward


def func(x):
    x = torch.nn.functional.relu(x)
    x = torch.add(x, x)
    x = torch.sigmoid(x)
    x = torch.nn.functional.gelu(x)
    x = x * x
    return x


def func_with_custom_op(x):
    x = torch.nn.functional.relu(x)
    x = torch.add(x, x)
    x = my_sigmoid(x)
    x = torch.nn.functional.gelu(x)
    x = x * x
    return x


def copy_tensor(tensor):
    another_tensor = torch.rand_like(tensor, requires_grad=tensor.requires_grad)
    another_tensor.data.copy_(tensor.data)
    return another_tensor


print('-' * 10)
print('Ground truth')
input_tensor = torch.randn(10, device='cuda', requires_grad=True)
output = func(input_tensor)
output.sum().backward()
print(output)
print(input_tensor.grad)

# print how dynamo handles custom op. It turns out that dynamo simply break the graph when encountering custom op
print('-' * 10)
print('Dynamo on graph with custom op without compile')
func_with_custom_triton_op_print = torch.compile(func_with_custom_op, backend=print_backend)
input_tensor_dynamo = copy_tensor(input_tensor)
output = func_with_custom_triton_op_print(input_tensor_dynamo)
output.sum().backward()
print(output)
print(input_tensor_dynamo.grad)

print('-' * 10)
print('Inductor on graph without custom op')
func_with_custom_triton_op_print = torch.compile(func_with_custom_op, backend=print_backend)
input_tensor_inductor = copy_tensor(input_tensor)
output = func_with_custom_triton_op_print(input_tensor_inductor)
output.sum().backward()
print(output)
print(input_tensor_inductor.grad)

print('-' * 10)
print('Compile graph with custom op')
# compile manual op replacement with inductor backend
torch._dynamo.reset()
func_with_custom_triton_op_inductor = torch.compile(func_with_custom_op)
input_tensor1 = copy_tensor(input_tensor)
output = func_with_custom_triton_op_inductor(input_tensor1)
output.sum().backward()
print(output)
print(input_tensor1.grad)

print('-' * 10)
print('Print graph with replaced op')
# graph replacement backend. Make sure pattern is replaced
torch._dynamo.reset()

func_with_custom_triton_op_inductor_ = torch.compile(func, backend=replace_pattern_backend)
input_tensor2 = copy_tensor(input_tensor)
output = func_with_custom_triton_op_inductor_(input_tensor2)
print(output)
output.sum().backward()
print(input_tensor2.grad)

# graph replacement with inductor backend.
print('-' * 10)
print('Replacement with inductor')
torch._dynamo.reset()
input_tensor3 = copy_tensor(input_tensor)
func_with_custom_triton_op_inductor__ = torch.compile(func, backend=replace_pattern_backend_with_inductor)
print(input_tensor3.requires_grad)
output = func_with_custom_triton_op_inductor__(input_tensor3)
print(output)
output.sum().backward()
print(input_tensor3.grad)
