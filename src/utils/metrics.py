import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from src.utils.logger import setup_logger

# Perplexity TODO
# Out-Of-Vocabulary TODO
# Training and Inference time TODO
# Average tokens per word for at least 1MB of text (the tested text has to be different than the text used to train the tokenizer). TODO
# Number of words directly present in the dictionary TODO

class EvaluationMetrics:
    def __init__(self, model, tokenizer, output_dir=None, logger=None):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer
        
        self.output_dir = output_dir
        self.logger = logger
        if not self.logger:
            self.logger = setup_logger("Evaluation Metrics", output_dir)


    def calculate_mean_perplexity(self, dataloader):
        self.model.eval()
        self.logger.info("Calculating perplexity score.")
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("<pad>"))
        total_tokens, total_loss = 0, 0.0

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Perplexity score"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

        mean_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(mean_loss))
        self.logger.info(f"Finished calculating perplexity. Perplexity score: {perplexity:.4f}")
        
        return perplexity
    
    def calculate_word_level_perplexity(self, dataloader):
        self.model.eval()
        self.logger.info("Calculating word-level perplexity.")
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("<pad>"))
        total_words, total_loss = 0, 0.0

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Word-level PPL"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                text = self.tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
                words = text.strip().split()
                total_words += len(words)
                total_loss += loss.item() * len(words)

        mean_loss = total_loss / total_words
        perplexity = torch.exp(torch.tensor(mean_loss))
        self.logger.info(f"Word-level perplexity: {perplexity:.4f}")
        return perplexity
    
    def calculate_character_level_perplexity(self, dataloader):
        self.model.eval()
        self.logger.info("Calculating character-level perplexity.")
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id("<pad>"))
        total_chars, total_loss = 0, 0.0

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Character-level PPL"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                text = self.tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
                total_chars += len(text)
                total_loss += loss.item() * len(text)

        mean_loss = total_loss / total_chars
        perplexity = torch.exp(torch.tensor(mean_loss))
        self.logger.info(f"Character-level perplexity: {perplexity:.4f}")
        return perplexity

    
    def calculate_oov_rate(self, data_dir):
        texts = self._load_texts(data_dir)
        self.logger.info("Calculating OOV rate.")
        total_words, oov_words = 0, 0
        for text in tqdm(texts, desc="OOV"):
            words = text.strip().split()
            total_words += len(words)
            for word in words:
                if self.tokenizer.token_to_id(word) is None:
                    oov_words += 1
        oov_rate = (oov_words / total_words) * 100 if total_words > 0 else 0
        self.logger.info(f"OOV rate: {oov_rate:.2f}% ({oov_words}/{total_words})")
        return oov_rate

    def calculate_avg_tokens_per_word(self, data_dir):
        texts = self._load_texts(data_dir)
        self.logger.info("Calculating average tokens per word.")
        total_words, total_tokens = 0, 0
        for text in tqdm(texts, desc="Avg tokens/word"):
            words = text.strip().split()
            total_words += len(words)
            total_tokens += len(self.tokenizer.encode(text))
        avg_tokens = total_tokens / total_words if total_words > 0 else 0
        self.logger.info(f"Average tokens per word: {avg_tokens:.4f}")
        return avg_tokens

    def count_words_in_dictionary(self, data_dir):
        texts = self._load_texts(data_dir)
        self.logger.info("Counting words directly present in dictionary.")
        total_words, in_dict = 0, 0
        for text in tqdm(texts, desc="Words in dict"):
            words = text.strip().split()
            total_words += len(words)
            for word in words:
                token_id = self.tokenizer.token_to_id(word)
                if token_id is not None:
                    in_dict += 1
        percent_in_dict = (in_dict / total_words) * 100 if total_words > 0 else 0
        self.logger.info(f"Words in dictionary: {in_dict}/{total_words} ({percent_in_dict:.2f}%)")
        return in_dict, percent_in_dict
    
    def calculate_tokenizer_throughput(self, dataloader):
        self.model.eval()
        self.logger.info("Calculating tokenizer throughput.")
        total_tokens = 0
        total_time = 0.0

        with torch.no_grad():
            for x, _ in tqdm(dataloader, desc="Tokenizer throughput"):
                x = x.to(self.device)

                t0 = time.time()

                logits = self.model(x)
            
                elapsed = time.time() - t0

                total_tokens += x.numel()
                total_time += elapsed

        throughput = total_tokens / total_time if total_time > 0 else 0
        self.logger.info(f"Tokenizer throughput: {throughput:.2f} tokens/s")
        return throughput

    
    def _load_texts(self, data_dir):
        texts = []
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return texts