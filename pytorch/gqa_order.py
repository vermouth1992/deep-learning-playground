import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, \
    LlamaLinearScalingRotaryEmbedding, apply_rotary_pos_emb, Cache, repeat_kv


def my_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :, :].expand(batch, n_rep, num_key_value_heads, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MyLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, gqa_order=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.gqa_order = gqa_order

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        print('Start')
        print(query_states)
        print(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.gqa_order == 1:
            key_states = my_repeat_kv(key_states, self.num_key_value_groups)
            value_states = my_repeat_kv(value_states, self.num_key_value_groups)
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        print('After repeat')
        print(query_states)
        print(key_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        print('Before project')
        print(attn_output)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        print('After project')
        print(attn_output)

        return attn_output, attn_weights, past_key_value


def reorder_q(weight, num_attention_heads, num_key_value_heads):
    hidden_size = weight.shape[0]
    weight = weight.view(num_key_value_heads, num_attention_heads // num_key_value_heads, hidden_size // num_attention_heads, hidden_size)
    weight = weight.transpose(0, 1).contiguous()
    weight = weight.view(hidden_size, hidden_size).contiguous()
    return weight


def reorder_o(weight, num_attention_heads, num_key_value_heads):
    hidden_size = weight.shape[0]
    weight = weight.view(hidden_size, num_key_value_heads, num_attention_heads // num_key_value_heads, hidden_size // num_attention_heads)
    weight = weight.transpose(1, 2).contiguous()
    weight = weight.view(hidden_size, hidden_size).contiguous()
    return weight


def order_state_dict(state_dict, num_attention_heads, num_key_value_heads):
    state_dict['q_proj.weight'] = reorder_q(state_dict['q_proj.weight'], num_attention_heads, num_key_value_heads)
    state_dict['o_proj.weight'] = reorder_o(state_dict['o_proj.weight'], num_attention_heads, num_key_value_heads)
    return state_dict


if __name__ == '__main__':
    config = LlamaConfig(vocab_size=16,
                         hidden_size=256,
                         intermediate_size=8,
                         num_hidden_layers=2,
                         num_attention_heads=64,
                         num_key_value_heads=4)

    torch.use_deterministic_algorithms(True)

    llama_attention = MyLlamaAttention(config=config, gqa_order=0)
    my_llama_attention = MyLlamaAttention(config=config, gqa_order=1)

    my_llama_attention.load_state_dict(order_state_dict(llama_attention.state_dict(), config.num_attention_heads,
                                                        config.num_key_value_heads))

    batch_size = 4
    seqlen = 16

    hidden_states = torch.randn(size=(batch_size, seqlen, config.hidden_size))
    attention_mask = torch.ones(size=(batch_size, seqlen), dtype=torch.int64)
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

    llama_output = llama_attention(hidden_states, None, position_ids)[0]
    my_llama_output = my_llama_attention(hidden_states, None, position_ids)[0]

    print(torch.all(torch.eq(llama_output, my_llama_output)))

    print(torch.max(torch.abs(llama_output - my_llama_output)))