
import math
import torch
import torch.nn as nn

vocab_size = 32 * 1024  # 32K words vocabulary (word -> embed)
embedding_dim = 4096    # 4096 dimension embedding (dmodel)
max_seq_length = 2048   # 2048 maximum input tokens

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


hidden_size = 4 * embedding_dim
ffn_linear1 = nn.Linear(embedding_dim, hidden_size)
ffn_linear2 = nn.Linear(hidden_size, embedding_dim)
ffn_act = nn.ReLU()

def feed_forward_naive(input_embd: torch.Tensor):
    hidden_states = ffn_act(ffn_linear1(input_embd))
    return ffn_linear2(hidden_states)


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


input_seq_length = 16 # 16 tokens
input_embd = torch.rand(input_seq_length, embedding_dim)

positional_encoding = build_absolute_positional_encoding_naive()
input_embd += positional_encoding[:input_embd.size(0)]

mh_attn_out = multi_head_attention_naive(input_embd, num_heads=32)
ffn_out = feed_forward_naive(mh_attn_out)