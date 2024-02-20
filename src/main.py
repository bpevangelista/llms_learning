import time, torch

import utils
from model_attention_naive import AttentionNaive
from model_attention_modern import AttentionXformers, AttentionSdpa
from model_blocks import LayerNorm

torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda:0')

# Transformer parameters
vocab_size = 32 * 1024  # 32K words/tokens embeddings in vocabulary
max_seq_length = 2048  # 2048 maximum input tokens
embedding_dim = 4096  # OPT 512, Llama2 4096 embedding dimension (dmodel)
num_heads = 32  # 16 or 32

mha_naive = AttentionNaive(embedding_dim, num_heads)
mha_xformers = AttentionXformers(embedding_dim, num_heads)
mha_sdpa = AttentionSdpa(embedding_dim, num_heads)
norm_layer = LayerNorm(embedding_dim)


def test_attn(hidden_states: torch.Tensor, attention_obj: any, layers: int, steps: int, title: str | None):
    memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f'\n{title:16} START Peak CUDA Mem: {memory_usage:.2f} MB')

    start = time.perf_counter()
    hidden = None
    for step in range(steps):
        hidden = hidden_states.clone()
        for layer in range(layers):
            hidden = attention_obj(norm_layer(hidden), None) + hidden
    end = time.perf_counter()

    memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f'{title:16} END   Peak CUDA Mem: {memory_usage:.2f} MB')
    print(f'{title:16}       Elapsed {(end - start) * 1000:.2f}ms')
    return hidden


def main():
    #model_name = 'facebook/opt-350m'
    model_name = 'mistralai/Mistral-7B-v0.1'
    tokenizer = utils.get_tokenizer(model_name)
    vocab_embedding = utils.get_vocab_embed(model_name)
    prompts = utils.get_prompts()[:1]

    tokens = tokenizer(prompts, padding=True, return_tensors='pt')
    hidden_states = vocab_embedding(tokens.input_ids)

    out1 = test_attn(hidden_states, mha_naive, 32, 100, 'NAIVE')
    out2 = test_attn(hidden_states, mha_xformers, 32, 100, 'XFORMERS')
    out3 = test_attn(hidden_states, mha_sdpa, 32, 100, 'SDPA')

    mse1 = torch.mean((out1 - out2) ** 2)
    mse2 = torch.mean((out1 - out3) ** 2)
    mse3 = torch.mean((out2 - out3) ** 2)
    print(f'\n{mse1}, {mse2}, {mse3}')


main()
