from datasets import load_from_disk
from tokenizers import Tokenizer
from typing import Dict, List, Any

class HFDatasetProcessor:
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.tokenizer.enable_truncation(max_length=max_seq_len+1)
        self.tokenizer.enable_padding(
            direction='right',
            pad_id=tokenizer.token_to_id("<pad>"), 
            length=max_seq_len + 1
        )
        self.dataset = load_from_disk(data_path)
        self.tokenized_data = self._process_dataset()

    def _tokenize_and_prepare(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        tokenized_output = self.tokenizer.encode_batch(
            examples["text"],
            add_special_tokens=True,
        )

        input_ids = [encoding.ids for encoding in tokenized_output]
        
        labels = [ids[1:] for ids in input_ids]
        input_ids = [ids[:-1] for ids in input_ids]
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _process_dataset(self):
        tokenized_dataset = self.dataset.map(
            self._tokenize_and_prepare,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )

        tokenized_dataset.set_format(type="torch", columns=['input_ids', 'labels'])
        return tokenized_dataset

    def get_data(self):
        return self.tokenized_data