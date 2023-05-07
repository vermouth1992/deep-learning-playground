import torch
import torch._dynamo
import torch._inductor.compile_fx

def add(a, b):
    return a + b


a = torch.randn(10)
b = torch.randn(10)

graph, guard = torch._dynamo.export(add, a, b)
out = torch._inductor.compile_fx.compile_fx_aot(graph, example_inputs_=[a, b])
