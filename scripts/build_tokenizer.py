from src.data.tokenizer.bpe_tokenizer_builder import BPETokenizerBuilder
from src.data.tokenizer.whitespace_tokenizer_builder import WhitespaceTokenizerBuilder
from src.utils.tokenizer_utils import download_pretrained_tokenizer

# GENERAL
build_tokenizers = {
    "BPE": True,
    "Pretrained": True,
    "Whitespace": True
}

TRAIN_DATA_DIR = "data/base"
TOKENIZERS_OUTPUT_DIR = "tokenizers"
VOCAB_SIZE = 50_000
MIN_FREQUENCY = 2

# BPE from training data
BPE_FILE_NAME = "bpe.json"

# Whitespace from training data
WS_FILE_NAME = "whitespace.json"

# Pretrained
PRETRAINED_URL = "https://huggingface.co/sdadas/polish-splade/resolve/main/tokenizer.json"
PRETRAINED_FILE_NAME = "polish-splade.json"



if __name__ == "__main__":
    if build_tokenizers["BPE"]:
        bpe_builder = BPETokenizerBuilder(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY
        )

        bpe_builder.build_from_directory(
            data_dir=TRAIN_DATA_DIR,
            output_dir=TOKENIZERS_OUTPUT_DIR,
            output_name=BPE_FILE_NAME
        )
    
    if build_tokenizers["Pretrained"]:
        download_pretrained_tokenizer(TOKENIZERS_OUTPUT_DIR, PRETRAINED_FILE_NAME, PRETRAINED_URL)

    if build_tokenizers["Whitespace"]:
        ws_builder = WhitespaceTokenizerBuilder(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY
        )

        ws_builder.build_from_directory(
            data_dir=TRAIN_DATA_DIR,
            output_dir=TOKENIZERS_OUTPUT_DIR,
            output_name=WS_FILE_NAME
        )