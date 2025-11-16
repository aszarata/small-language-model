from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from src.utils.metrics import EvaluationMetrics
from src.data.text_dataset import TextDataset
from src.utils.model_utils import load_model
from src.utils.logger import setup_logger

MODEL_DIR = "models/model-whitespace/checkpoint_6"
TOKENIZER_PATH = "models/model-whitespace/tokenizer.json"
EVALUATION_RESULTS_DIR = "evaluation-results/model-whitespace"
DATA_DIR = "data/base/test"
MAX_SEQ_LEN = 128
BATCH_SIZE = 64

def evaluate_metrics(model_dir, tokenizer_path, results_dir, data_dir, logger=None):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(model_dir)

    # Data
    dataset = TextDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_len=MAX_SEQ_LEN
    )

    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    evaluation_metrics = EvaluationMetrics(
        model=model,
        tokenizer=tokenizer,
        output_dir=results_dir,
        logger=logger
    )

    throughput = evaluation_metrics.calculate_tokenizer_throughput(dataloader=test_loader)
    token_lvl_perplexity = evaluation_metrics.calculate_mean_perplexity(dataloader=test_loader)
    word_lvl_perplexity = evaluation_metrics.calculate_word_level_perplexity(dataloader=test_loader)
    char_lvl_perplexity = evaluation_metrics.calculate_character_level_perplexity(dataloader=test_loader)
    oov = evaluation_metrics.calculate_oov_rate(data_dir)
    avg_tokens_per_word = evaluation_metrics.calculate_avg_tokens_per_word(data_dir)
    words_in_dict = evaluation_metrics.count_words_in_dictionary(data_dir)

if __name__ == "__main__":
    logger = setup_logger("Evaluation Metrics", log_dir=EVALUATION_RESULTS_DIR + "/logs")

    evaluate_metrics(
        model_dir=MODEL_DIR,
        tokenizer_path=TOKENIZER_PATH,
        results_dir=EVALUATION_RESULTS_DIR,
        data_dir=DATA_DIR,
        logger = logger
    )
