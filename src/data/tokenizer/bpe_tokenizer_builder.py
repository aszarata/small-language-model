import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from src.data.tokenizer.tokenizer_builder import TokenizerBuilder

class BPETokenizerBuilder(TokenizerBuilder):
    def __init__(self, vocab_size=40000, min_frequency=2):
        super().__init__(vocab_size=vocab_size, min_frequency=min_frequency)
    
    def build(self, paths, output_path):
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

        tokenizer.train(paths, trainer)
        tokenizer.save(output_path)