import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')
        ]
        self.tokens = []

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                tokenized_text = tokenizer.encode(text).ids
                self.tokens.extend(tokenized_text)
        
        self.tokens = torch.tensor(self.tokens)

        total_len = (len(self.tokens) // seq_len) * seq_len
        self.tokens = self.tokens[:total_len + 1]

    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return {
            'input_ids': x,
            'labels': y
        }