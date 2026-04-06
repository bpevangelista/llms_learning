import math
import torch
import torch.nn as nn
import model_utils as utils


class AttentionNaive(nn.Module):
    def __init__(self, embedding_dim: int, num_qkv_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_qkv_heads
        self.head_size = embedding_dim // num_qkv_heads

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor = None):
        # [seq_len, hidden_dim] (no batching)
        seq_length = hidden_states.size(0)

        q1 = self.q_proj(hidden_states)
        k1 = self.k_proj(hidden_states)
        v1 = self.v_proj(hidden_states)

        # We work on per-head sequences
        # Rearrange as [seq_length, num_heads, head_size] (no batching)
        q2 = q1.view(seq_length, self.num_heads, self.head_size).transpose(0, 1)
        k2 = k1.view(seq_length, self.num_heads, self.head_size).transpose(0, 1)
        v2 = v1.view(seq_length, self.num_heads, self.head_size).transpose(0, 1)

        # q * k_transposed / sqrt(dk)
        dk = self.head_size
        qk = q2.matmul(k2.transpose(-2, -1)) / math.sqrt(dk)

        # Causal mask
        inf_full = torch.full((seq_length, seq_length), float('-inf'), device=qk.device)
        causal_mask = torch.triu(inf_full, diagonal=1)
        qk = qk + causal_mask

        # out = softmax(qk) * v (no out projection for now)
        mh_attn = nn.functional.softmax(qk, dim=-1)
        mh_attn_out = mh_attn.matmul(v2)

        # Rearrange data back as [seq_length, embedding_dim]
        mh_attn_out = mh_attn_out.transpose(0, 1).reshape(seq_length, self.embedding_dim)
        mh_attn_out = self.out_proj(mh_attn_out)
        return mh_attn_out


class LayerNormNaive(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attn_norm_layer = nn.LayerNorm(embedding_dim)  # Learnable

    def forward(self, hidden_states: torch.Tensor):
        return self.attn_norm_layer(hidden_states)


class MLPNaive(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        intermediate_size = 4 * embedding_dim
        self.ffn_linear1 = nn.Linear(embedding_dim, intermediate_size)
        self.ffn_linear2 = nn.Linear(intermediate_size, embedding_dim)
        self.ffn_act = nn.ReLU()
        self.ffn_drop = nn.Dropout(0.0)  # Training-only, discard X% to avoid over fit

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.ffn_linear1(hidden_states)
        hidden_states = self.ffn_drop(self.ffn_act(hidden_states))
        hidden_states = self.ffn_linear2(hidden_states)
        return hidden_states


class DecoderLayerNaive(nn.Module):
    def __init__(self, embedding_dim: int, num_qkv_heads: int):
        super().__init__()
        self.multi_head_attention = AttentionNaive(embedding_dim, num_qkv_heads)
        self.mlp = MLPNaive(embedding_dim)
        self.post_attention_norm = LayerNormNaive(embedding_dim)
        self.post_feed_forward_norm = LayerNormNaive(embedding_dim)


class TransformerConfig:
    def __init__(self, vocab_size: int, num_hidden_layers: int, max_seq_length: int, embedding_dim: int,
                 num_qkv_heads: int, bos_token_id: int = 1, eos_token_id: int = 2, pad_token_id: int = 2):
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_qkv_heads = num_qkv_heads
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.vocab_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.positional_encoding = utils.build_absolute_positional_encoding_naive(config.embedding_dim, config.max_seq_length)
        self.decoder = nn.ModuleList(
            [DecoderLayerNaive(config.embedding_dim, config.num_qkv_heads) for layer in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor):
        has_finished = False

        while not has_finished:
            hidden_states = self.vocab_embedding(input_ids)
            # [seq_len, hidden_dim] (no batching)
            seq_length = hidden_states.size(0)
            # On naive, applies positional_encoding to Q, K, V
            hidden_states = hidden_states + self.positional_encoding[:seq_length]

            for layer in self.decoder:
                # On naive, num_heads is the same across Q, K, V
                residual = hidden_states
                hidden_states = layer.multi_head_attention(hidden_states)
                hidden_states = layer.post_attention_norm(residual + hidden_states)

                residual = hidden_states
                hidden_states = layer.mlp(hidden_states)
                hidden_states = layer.post_feed_forward_norm(residual + hidden_states)

            logits = self.lm_head(hidden_states)

            # Greedy sampling over last/most-recent next-token
            next_token_id = torch.argmax(logits[-1, :]).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id])
            print(f'input_ids shape: {input_ids.shape}')
            # print(f'{input_ids}')

            if len(input_ids) >= self.config.max_seq_length or next_token_id[0] == self.config.eos_token_id:
                has_finished = True


def main():
    # Transformer parameters
    vocab_size = 32 * 1024  # 32K words/tokens vocabulary size
    embedding_dim = 4096  # 4096 token embedding dimension (dmodel)
    max_seq_length = 2048  # 2048 max input+output token sequence length
    num_qkv_heads = 32  # 32 constant QKV heads (128 embd_size each)
    # num_decoder_layers = 32
    num_decoder_layers = 1

    tokenizer = utils.get_tokenizer('facebook/opt-350m')
    config = TransformerConfig(vocab_size, num_decoder_layers, max_seq_length, embedding_dim, num_qkv_heads,
                               eos_token_id=tokenizer.eos_token_id)
    llm = TransformerDecoder(config)

    prompt = 'Which fruits do you like?'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').squeeze(0)  # remove batching
    print(f'input_ids shape: {input_ids.shape}')

    llm(input_ids)


main()
