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
    
    def build(self, output_path, paths=None, dataset=None, text_column="input_ids"):
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

        if dataset:
            def batch_iterator(batch_size=1000):
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    yield batch[text_column]

            tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

        elif paths:
            tokenizer.train(paths, trainer)
        
        else:
            raise Exception("No paths or dataset specified to train tokenizer.")
        
        tokenizer.save(output_path)