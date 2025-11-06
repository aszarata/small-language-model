from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import os

def build_byte_level_bpe_tokenizer(data_dir, vocab_size=40000, min_frequency=2, output_dir="tokenizers"):
    paths = [str(p) for p in Path(data_dir).rglob("*.txt")]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer.train(paths, trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(f"{output_dir}/tokenizer.json")

    return output_dir