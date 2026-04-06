import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_prompts() -> list[str]:
    with open('prompts.bin', 'rb') as file:
        return pickle.load(file)


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_vocab_embed(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else model.model.decoder.embed_tokens


def xavier_normal_tensor(in_dim: int, out_dim: int) -> torch.Tensor:
    x = torch.empty(in_dim, out_dim)
    torch.nn.init.xavier_normal_(x)
    return x


def build_absolute_positional_encoding_naive(embedding_dim: int, max_seq_length: int) -> torch.Tensor:
    positions = torch.arange(max_seq_length)  # [0,1,2, ... max_seq_length]
    positions = positions.unsqueeze(1)  # [0,1,2, ... max_seq_length][]
    embeddings = torch.arange(embedding_dim)  # [0,1,2, ... embedding_dim]

    # pos/10000^(2i/dmodel) — embeddings // 2 so sin/cos share pairs
    angle = positions / torch.pow(10000, (2 * (embeddings // 2) / embedding_dim))

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

    nn.init.xavier_normal_(q_proj)
    nn.init.xavier_normal_(k_proj)
    nn.init.xavier_normal_(v_proj)
    return q_proj, k_proj, v_proj
