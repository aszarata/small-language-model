from dataclasses import dataclass, asdict
import json

@dataclass
class LanguageModelConfig:
    vocab_size: int  
    n_position: int 
    embed_dim: int 
    hidden_dim: int 
    num_heads: int 
    n_layers: int 
    dropout_prob: float = 0.1
    n_classes: int = None

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**json.load(f))
        
    def __str__(self):
        config_dict = asdict(self)
        lines = ["LanguageModel configuration:"]
        for key, value in config_dict.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
