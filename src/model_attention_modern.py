import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops

import utils


def build_rotary_positional_embedding(embedding_dim: int, max_seq_length: int) -> torch.Tensor:
    return torch.zeros(max_seq_length, embedding_dim)  # dummy


def apply_rotary_positional_embedding(query: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor):
    return query, value


def build_qkv_proj(embedding_dim: int) -> torch.Tensor:
    q_proj = torch.empty(embedding_dim, embedding_dim)
    k_proj = torch.empty(embedding_dim, embedding_dim)
    v_proj = torch.empty(embedding_dim, embedding_dim)

    torch.nn.init.xavier_normal_(q_proj)
    torch.nn.init.xavier_normal_(k_proj)
    torch.nn.init.xavier_normal_(v_proj)
    qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0)
    return qkv_proj


class AttentionModern(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.rope = build_rotary_positional_embedding(embedding_dim, 4096)
        self.qkv_proj = build_qkv_proj(embedding_dim)
        self.out_proj = utils.xavier_normal_tensor(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

    def _proj_and_pos_encode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        # QKV Projection
        qkv = F.linear(hidden_states, self.qkv_proj)
        query, key, value = qkv.split([self.embedding_dim, self.embedding_dim, self.embedding_dim], dim=-1)

        # Relative Rotary Position Embedding
        query, key = apply_rotary_positional_embedding(query, key, position_ids)

        # Note: Non KV-Cached Step

        # Multi-Head Reshape. [batch, seq, embd] -> [batch*seq, num_heads, head_size]
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)
        return query, key, value


class AttentionXformers(AttentionModern):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__(embedding_dim, num_heads)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        batches = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        query, key, value = self._proj_and_pos_encode(hidden_states, position_ids)

        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([seq_length] * batches)
        mh_attn_out = xformers.ops.memory_efficient_attention_forward(
            query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), attn_bias=attn_bias,
        )

        mh_attn_out = mh_attn_out.reshape(batches, seq_length, self.embedding_dim)
        mh_attn_out = F.linear(mh_attn_out, self.out_proj)
        return mh_attn_out


class AttentionSdpa(AttentionModern):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__(embedding_dim, num_heads)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        batches = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        query, key, value = self._proj_and_pos_encode(hidden_states, position_ids)

        mh_attn_out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, is_causal=True,
        )

        mh_attn_out = mh_attn_out.reshape(batches, seq_length, self.embedding_dim)
        mh_attn_out = F.linear(mh_attn_out, self.out_proj)
        return mh_attn_out
