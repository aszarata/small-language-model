import torch
import torch.nn as nn
from .transformer_model import LanguageModel
from .model_config import LanguageModelConfig

class Classifier(LanguageModel):
    def __init__(self, n_classes: int, config: LanguageModelConfig, pad_id: int = 1):
        super().__init__(config)
        self.classification_head = nn.Linear(config.embed_dim, n_classes)
        self.pad_id = pad_id

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x_emb = self.token_embedding(x) + self.positional_encoding(positions)
        x_emb = self.transformer_blocks(x_emb)
        x_emb = self.layer_norm(x_emb)

        mask = (x != self.pad_id).float().unsqueeze(-1)

        summed = (x_emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        sentence_repr = summed / lengths

        return self.classification_head(sentence_repr)