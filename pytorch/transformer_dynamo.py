import torch.nn.functional as F
import torch

import transformers

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
num_heads = 3
seq_len = 10
embed_dim = 10

query = torch.randn(batch_size, num_heads, seq_len, embed_dim)
key = torch.randn(batch_size, num_heads, seq_len, embed_dim)
value = torch.randn(batch_size, num_heads, seq_len, embed_dim)
attn_bias = torch.randn(batch_size, num_heads, seq_len, seq_len)

output = scale_dot_product(query, key, value, p=0., attn_bias=attn_bias)
output_2 = F.scaled_dot_product_attention(query, key, value, attn_bias, 0., False)

torch.testing.assert_close(output_2, output)
