import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


import jax.numpy as jnp
import jax.lax

# torch.gather equivalent
def jax_gather(x: jnp.ndarray, indices: jnp.ndarray, axis: int) -> jnp.ndarray:
    complete_indices = jnp.array(jnp.where(indices > -1))
    complete_indices = complete_indices.at[axis].set(jnp.reshape(indices, [-1]))
    flat_ind = jnp.ravel_multi_index(tuple(complete_indices), x.shape)
    return jnp.reshape(jax.lax.gather(jnp.reshape(x, [-1]), flat_ind), indices.shape)


def jax_rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_forward_jax(q, k, cos: jnp.ndarray, sin, position_ids):
    gather_indices = jnp.expand_dims(position_ids, axis=(1, 3))
    gather_indices = jnp.tile(gather_indices, jnp.asarray((1, cos.shape[0], 1, cos.shape[3])))
    cos = jnp.tile(cos, (gather_indices.shape[0], 1, 1, 1))
    sin = jnp.tile(sin, (gather_indices.shape[0], 1, 1, 1))
    cos = jax_gather(cos, gather_indices, axis=2)
    sin = jax_gather(sin, gather_indices, axis=2)
    q_embed = (q * cos) + (jax_rotate_half(q) * sin)
    k_embed = (k * cos) + (jax_rotate_half(k) * sin)
    return q_embed, k_embed


def print_compile_fn(fx_module, args):
    print(fx_module)
    return fx_module


from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack


def torch_tensor_to_jax(x_torch):
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax


if __name__ == '__main__':
    batch_size = 1
    seq_len = 1
    num_heads = 1
    size_per_dim = 2
    device = 'cuda'
    q = torch.randn(batch_size, num_heads, seq_len, size_per_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, size_per_dim, device=device)
    cos = torch.randn(1, 1, seq_len, size_per_dim, device=device)
    sin = torch.randn(1, 1, seq_len, size_per_dim, device=device)
    position_ids = torch.randint(0, seq_len, size=(batch_size, seq_len), device=device)

    out = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    out_jax = apply_rotary_pos_emb_forward_jax(torch_tensor_to_jax(q),
                                               torch_tensor_to_jax(k),
                                               torch_tensor_to_jax(cos),
                                               torch_tensor_to_jax(sin),
                                               torch_tensor_to_jax(position_ids))
    #
    # aot_fn = aot_function(fn=apply_rotary_pos_emb, fw_compiler=print_compile_fn, bw_compiler=print_compile_fn)
    # out = aot_fn(q, k, cos, sin, position_ids)
