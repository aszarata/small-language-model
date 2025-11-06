from src.utils.tokenizer_utils import build_byte_level_bpe_tokenizer

DATA_DIR = "data/base"
VOCAB_SIZE = 40000
MIN_FREQUENCY = 2
OUTPUT_DIR = "tokenizers/bpe-test-40000"


if __name__ == "__main__":
    build_byte_level_bpe_tokenizer(DATA_DIR, VOCAB_SIZE, MIN_FREQUENCY, OUTPUT_DIR)