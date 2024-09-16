import gc, time, torch

from model_transformer_naive import AttentionNaive
from attention_modern import AttentionXformers, AttentionSdpa
import model_utils as utils

embedding_dim = 4096  # OPT 512, Llama2 4096 embedding dimension (dmodel)
num_qkv_heads = 32
num_hidden_layers = 32
num_steps = 128
norm_layer = torch.nn.LayerNorm(embedding_dim)

torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda:0')


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def test_attn(hidden_states: torch.Tensor, attentions: list[any], steps: int, title: str | None):
    memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f'\n{title:16} START Peak CUDA Mem: {memory_usage:.2f} MB')

    start = time.perf_counter()
    hidden = None
    for step in range(steps):
        hidden = hidden_states.clone()
        for attention in attentions:
            hidden = attention(norm_layer(hidden), None) + hidden
    end = time.perf_counter()

    memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f'{title:16} END   Peak CUDA Mem: {memory_usage:.2f} MB')
    print(f'{title:16}       Elapsed {(end - start) * 1000:.2f}ms')
    return hidden


def main():
    # model_name = 'facebook/opt-350m'
    model_name = 'mistralai/Mistral-7B-v0.1'
    tokenizer = utils.get_tokenizer(model_name)
    vocab_embedding = utils.get_vocab_embed(model_name)
    prompts = utils.get_prompts()[:1]

    tokens = tokenizer(prompts, padding=True, return_tensors='pt')
    hidden_states = vocab_embedding(tokens.input_ids)

    out1 = hidden_states.clone()
    out2 = hidden_states.clone()
    out3 = hidden_states.clone()

    free_mem()
    mha_naive = [AttentionNaive(embedding_dim, num_qkv_heads) for _ in range(num_hidden_layers)]
    out1 = test_attn(hidden_states, mha_naive, num_steps, 'NAIVE')
    mha_naive.clear()

    free_mem()
    mha_xformers = [AttentionXformers(embedding_dim, num_qkv_heads) for _ in range(num_hidden_layers)]
    out2 = test_attn(hidden_states, mha_xformers, num_steps, 'XFORMERS')
    mha_xformers.clear()

    free_mem()
    mha_sdpa = [AttentionSdpa(embedding_dim, num_qkv_heads) for _ in range(num_hidden_layers)]
    out3 = test_attn(hidden_states, mha_sdpa, num_steps, 'SDPA')
    mha_sdpa.clear()

    # Note MSE will vary due to different positional encoding
    mse1 = torch.mean((out1 - out2) ** 2)
    mse2 = torch.mean((out1 - out3) ** 2)
    mse3 = torch.mean((out2 - out3) ** 2)
    print(f'\nMSEs {mse1}, {mse2}, {mse3}')


print(
    f'Torch ({torch.__version__}), CPUs: {torch.get_num_threads()}, ' +
    f'CUDA_GPUs: {torch.cuda.device_count()}, ' +
    f'MPS_GPU: {torch.backends.mps.is_available()}\n')

main()
