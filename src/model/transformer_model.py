import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .model_config import LanguageModelConfig

class LanguageModel(nn.Module):
    def __init__(self, config: LanguageModelConfig):
        super().__init__()
        self.config = config

        self.positional_encoding = nn.Embedding(config.n_position, config.embed_dim)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        blocks = [
            TransformerBlock(
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout_prob=config.dropout_prob,
            ) for _ in range(config.n_layers)
        ]

        self.transformer_blocks = nn.Sequential(*blocks)

        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.vocab_size)


    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.positional_encoding(positions)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

