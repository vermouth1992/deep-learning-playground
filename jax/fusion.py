import torch


def timed_cuda(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output = fn()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x + x


import jax
import jax.numpy as jnp


@jax.jit
def bloom_gelu_forward_jax(x: jax.Array) -> jax.Array:
    return x * 0.5 * (1.0 + jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_forward_jax_no_jit(x: jax.Array) -> jax.Array:
    return x * 0.5 * (1.0 + jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


from jax2torch import tree_j2t, tree_t2j, jax2torch

# def bloom_gelu_forward_jax_to_torch(x):
#     args = tree_t2j((x,))
#     # y_, ctx.fun_vjp = jax.vjp(fn, *args)
#     y_ = bloom_gelu_forward_jax(*args)
#     return tree_j2t(y_)


#
bloom_gelu_forward_jax_to_torch = jax2torch(bloom_gelu_forward_jax)

if __name__ == '__main__':
    a = torch.randn(500, 500, dtype=torch.float16, device='cuda', requires_grad=True)
    a_prime = a.clone().detach().requires_grad_()
    key = jax.random.PRNGKey(1021)
    a_jax = jax.random.normal(key=key, shape=(500, 500), dtype=jnp.float16)

    #
    # def jax_fn():
    #     output = bloom_gelu_forward_jax(a_jax)
    #     output.mean().backward()


    # def jax_no_jit_fn():
    #     output = bloom_gelu_forward_jax_no_jit(a_jax)


    def jax_torch_fn():
        output = bloom_gelu_forward_jax_to_torch(a)
        output.mean().backward()


    def torch_fn():
        output = bloom_gelu_forward(a_prime)
        output.mean().backward()


    for _ in range(10):
        jax_torch_fn()
        torch_fn()
        # jax_fn()
        # jax_no_jit_fn()

    torch.cuda.synchronize()

    # for i in range(10):
    #     _, jax_time = timed_cuda(jax_no_jit_fn)
    #     print(f'Iteration {i}, time {jax_time}')
    #
    # for i in range(10):
    #     _, jax_time = timed_cuda(jax_fn)
    #     print(f'Iteration {i}, time {jax_time}')

    for i in range(10):
        _, jax_time = timed_cuda(jax_torch_fn)
        print(f'Iteration {i}, time {jax_time}')

    for i in range(10):
        _, torch_time = timed_cuda(torch_fn)
        print(f'Iteration {i}, time {torch_time}')
