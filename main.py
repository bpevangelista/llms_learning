
import math
import torch
import torch.nn as nn


# Transformer parameters
vocab_size = 32 * 1024  # 32K words/tokens embeddings in vocabulary
embedding_dim = 4096    # 4096 embedding dimension (dmodel)
max_seq_length = 2048   # 2048 maximum input tokens

def build_absolute_positional_encoding_naive():
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


# q, k, v – layers were learned during training
q = nn.Linear(embedding_dim, embedding_dim, bias=False)
k = nn.Linear(embedding_dim, embedding_dim, bias=False)
v = nn.Linear(embedding_dim, embedding_dim, bias=False)

def multi_head_attention_naive(input_embd: torch.Tensor,
                               num_heads: int) -> torch.Tensor:
    q1 = q(input_embd)
    k1 = k(input_embd)
    v1 = v(input_embd)

    seq_length = input_embd.size(0)
    head_length = embedding_dim // num_heads

    # We work on per-head sequences (no batching for now)
    # Rearrange data as [num_heads, seq_length, head_length]
    q2 = q1.view(seq_length, num_heads, head_length).transpose(0, 1)
    k2 = k1.view(seq_length, num_heads, head_length).transpose(0, 1)
    v2 = v1.view(seq_length, num_heads, head_length).transpose(0, 1)

    # q * k_transposed / sqrt(dk)
    dk = head_length
    qk = q2.matmul(k2.transpose(1, 2)) / math.sqrt(dk)

    # out = softmax(qk) * v (no out projection for now)
    mh_attn = nn.functional.softmax(qk, dim=-1)
    mh_attn_out = mh_attn.matmul(v2)

    # Rearrange data back as [seq_length, embedding_dim]
    mh_attn_out = mh_attn_out.transpose(0, 1).reshape(seq_length, embedding_dim)
    return mh_attn_out


# Does mean and std normalization, then applies learnable weight and bias
norm_layer1 = nn.LayerNorm(embedding_dim)  # learnable
norm_layer2 = nn.LayerNorm(embedding_dim)  # learnable

def multi_head_attention_add_and_norm_naive(input_embd: torch.Tensor):
    return norm_layer1(input_embd)

def feed_forward_add_norm_naive(input_embd: torch.Tensor):
    return norm_layer2(input_embd)


ffn_linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
ffn_linear2 = nn.Linear(4 * embedding_dim, embedding_dim)
ffn_act = nn.ReLU()
ffn_drop = nn.Dropout(0.1)  # discard random 10% to avoid overfit

def feed_forward_naive(input_embd: torch.Tensor):
    hidden_states = ffn_linear1(input_embd)
    hidden_states = ffn_drop(ffn_act(hidden_states))
    hidden_states = ffn_linear2(hidden_states)
    return hidden_states