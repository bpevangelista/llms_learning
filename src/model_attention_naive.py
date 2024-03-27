import math
import torch

import torch.nn as nn
import torch.nn.functional as F

import utils


def build_absolute_positional_encoding_naive(embedding_dim: int, max_seq_length: int) -> torch.Tensor:
    positions = torch.arange(max_seq_length)  # [0,1,2, ... max_seq_length]
    positions = positions.unsqueeze(1)  # [0,1,2, ... max_seq_length][]
    embeddings = torch.arange(embedding_dim)  # [0,1,2, ... embedding_dim]

    # pos/10000^(2i/dmodel)
    angle = positions / torch.pow(10000, (2 * embeddings / embedding_dim))

    # PE(pos, 2i+0) = sin( pos/10000^(2i/dmodel) )
    # PE(pos, 2i+1) = cos( pos/10000^(2i/dmodel) )
    positional_encoding = torch.empty(max_seq_length, embedding_dim)
    positional_encoding[:, 0::2] = torch.sin(angle[:, 0::2])
    positional_encoding[:, 1::2] = torch.cos(angle[:, 1::2])
    return positional_encoding


def build_q_k_v_proj(embedding_dim: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    q_proj = torch.empty(embedding_dim, embedding_dim)
    k_proj = torch.empty(embedding_dim, embedding_dim)
    v_proj = torch.empty(embedding_dim, embedding_dim)

    torch.nn.init.xavier_normal_(q_proj)
    torch.nn.init.xavier_normal_(k_proj)
    torch.nn.init.xavier_normal_(v_proj)
    return q_proj, k_proj, v_proj


class AttentionNaive(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.pos_encoding = build_absolute_positional_encoding_naive(embedding_dim, 4096)
        self.q_proj, self.k_proj, self.v_proj = build_q_k_v_proj(embedding_dim)
        self.out_proj = utils.xavier_normal_tensor(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        batches = hidden_states.size(0)
        seq_length = hidden_states.size(1)

        hidden_states = hidden_states + self.pos_encoding[:seq_length]

        q1 = F.linear(hidden_states, self.q_proj)
        k1 = F.linear(hidden_states, self.k_proj)
        v1 = F.linear(hidden_states, self.v_proj)

        # We work on per-head sequences
        # Rearrange as [batches, seq_length, num_heads, head_size]
        q2 = q1.view(-1, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k2 = k1.view(-1, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v2 = v1.view(-1, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        # q * k_transposed / sqrt(dk)
        dk = self.head_size
        qk = q2.matmul(k2.transpose(-2, -1)) / math.sqrt(dk)

        # out = softmax(qk) * v (no out projection for now)
        mh_attn = nn.functional.softmax(qk, dim=-1)
        mh_attn_out = mh_attn.matmul(v2)

        # Rearrange data back as [seq_length, embedding_dim]
        mh_attn_out = mh_attn_out.transpose(1, 2).reshape(batches, seq_length, self.embedding_dim)
        mh_attn_out = F.linear(mh_attn_out, self.out_proj)
        return mh_attn_out
