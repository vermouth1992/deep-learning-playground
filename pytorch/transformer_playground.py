from torch import nn
import torch
import math
import torch.nn.functional as F


def print_shape(names, tensors):
    strings = []
    for name, tensor in zip(names, tensors):
        strings.append(f'{name}:{tensor.shape}')
    print(', '.join(strings))


def scaled_dot_product(q, k, v, mask=None):
    # q, k, v shape: # (batch_size, num_heads, seq_length, head_dim)
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_length, seq_length)
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
    values = torch.matmul(attention, v)  # (batch_size, num_heads, seq_length, head_dim)

    print_shape(names=['q', 'v', 'v', 'attn_logits', 'attention', 'values'],
                tensors=[q, k, v, attn_logits, attention, values])

    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        # x shape: (batch_size, seq_length, input_dim)

        batch_size, seq_length, _ = x.size()
        # obtain q,k,v from the input  (bs, seq_length, input_dim) -> (bs, seq_length, 3 * embed_dim)
        # head_dim = embed_dim // num_heads
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, num_heads, seq_length, head_dim)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)  # (batch_size, num_heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)  # (batch_size, seq_length, embed_dim)
        o = self.o_proj(values)  # (batch_size, seq_length, embed_dim)

        if return_attention:
            return o, attention
        else:
            return o


if __name__ == '__main__':
    input_dim = 5
    seq_length = 10
    batch_size = 2
    x = torch.randn(batch_size, seq_length, input_dim)
    model = MultiheadAttention(input_dim=input_dim, embed_dim=12, num_heads=3)
    output = model(x)
