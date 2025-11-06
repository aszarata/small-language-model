import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout_prob=0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.m_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.dropout = nn.Dropout(dropout_prob)


        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        y, _ = self.m_attention(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))
        y = self.dropout(y)
        
        x = self.layer_norm2(x + y)

        y = self.mlp(x)
        y = self.dropout(y)
        return x + y
