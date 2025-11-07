from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from src.data.tokenizer.tokenizer_builder import TokenizerBuilder

class WhitespaceTokenizerBuilder(TokenizerBuilder):
    def __init__(self, vocab_size=40000, min_frequency=2):
        super().__init__(vocab_size=vocab_size, min_frequency=min_frequency)
        
    
    def build(self, paths, output_path):
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
        )

        tokenizer.train(files=paths, trainer=trainer)
        tokenizer.save(output_path)