import pickle, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def get_prompts() -> list[str]:
    with open('prompts.bin', 'rb') as file:
        return pickle.load(file)


def xavier_normal_tensor(in_dim: int, out_dim: int) -> torch.Tensor:
    x = torch.empty(in_dim, out_dim)
    torch.nn.init.xavier_normal_(x)
    return x

# Transformer Weights - q, k, v are learned during training
