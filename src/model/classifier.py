import torch
import torch.nn as nn
from src.model.transformer_model import LanguageModel

class Classifier(nn.Module):
    def __init__(self, n_classes, language_model=None):
        super().__init__()
        self.language_model = language_model
        if not self.language_model:
            self.language_model = ...

        self.classifier = nn.Linear(self.language_model.config.embed_dim, n_classes)

    def forward(self, x):
        x = self.language_model(x)
        last_token = x[:, -1, :]
        return self.classifier(last_token)