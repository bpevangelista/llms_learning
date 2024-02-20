import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attn_norm_layer = nn.LayerNorm(embedding_dim)  # learnable

    def forward(self, hidden_states: torch.Tensor):
        return self.attn_norm_layer(hidden_states)


class MLP(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.ffn_linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.ffn_linear2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.ffn_act = nn.ReLU()
        self.ffn_drop = nn.Dropout(0.0)  # Training-only, discard X% to avoid over fit

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.ffn_linear1(hidden_states)
        hidden_states = self.ffn_drop(self.ffn_act(hidden_states))
        hidden_states = self.ffn_linear2(hidden_states)
        return hidden_states
