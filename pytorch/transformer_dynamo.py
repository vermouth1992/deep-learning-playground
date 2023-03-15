import triton
import torch.nn.functional as F
import torch


def backend(model: torch.fx.GraphModule, input):
    model.graph.print_tabular()
    return model


@torch.compile(backend=backend)
def scale_dot_product(query, key, value, p, attn_bias=None):
    """
    :param query: (batch_size, num_heads, seq_len, embed_dim)
    :param key: (batch_size, num_heads, seq_len, embed_dim)
    :param value: (batch_size, num_heads, seq_len, embed_dim)
    :param p: dropout probability
    :param attn_bias: attention_bias
    :return:
    """
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    return attn @ value


batch_size = 1
num_heads = 12
seq_len = 12
embed_dim = 12
dtype = torch.float16
device = 'cuda'

query = torch.randn(batch_size, num_heads, seq_len,
                    embed_dim, dtype=dtype, device=device)
key = torch.randn(batch_size, num_heads, seq_len,
                  embed_dim, dtype=dtype, device=device)
value = torch.randn(batch_size, num_heads, seq_len,
                    embed_dim, dtype=dtype, device=device)
attn_bias = torch.randn(batch_size, num_heads, seq_len,
                        seq_len, dtype=dtype, device=device)

attn_bias = None

output = scale_dot_product(query, key, value, p=0., attn_bias=attn_bias)
output_2 = F.scaled_dot_product_attention(
    query, key, value, attn_bias, 0., False)

try:
    torch.testing.assert_close(output_2, output, atol=1e-3, rtol=1e-2)
except Exception as e:
    print(e)


torch_time = triton.testing.do_bench(lambda: scale_dot_product(query, key, value, p=0., attn_bias=attn_bias),
                                     percentiles=None)
flash_time = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(
    query, key, value, attn_bias, 0., False), percentiles=None)

print(torch_time, flash_time)
