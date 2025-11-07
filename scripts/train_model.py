import yaml
import torch
from torch.utils.data import DataLoader
from src.model.model_config import LanguageModelConfig
from src.model.transformer_model import LanguageModel
from src.training.trainer import Trainer
from src.data.text_dataset import TextDataset
from src.utils.model_utils import load_model
from tokenizers import Tokenizer

CONFIG_FILE = "configs/base.yaml"
TOKENIZER_PATH = "tokenizers/bpe-test-40000/tokenizer.json"
DATA_DIR = "data/base/train"
SAVE_MODEL_DIR = "models/base-1"
MODEL_PATH = None

def train(config_file, tokenizer_path, data_dir, save_model_dir, model_path=None):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Model
    if model_path:
        model = load_model(model_path)
    else:
        model_config = LanguageModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            n_position=config["model"]["n_position"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_heads=config["model"]["num_heads"],
            n_layers=config["model"]["n_layers"],
            dropout_prob=config["model"]["dropout_prob"]
        )
        model = LanguageModel(model_config)

    # Data
    dataset = TextDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        seq_len=config["training"]["max_seq_len"]
    )

    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"])
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion
    )
    
    trainer.fit(
        train_loader=train_loader,
        epochs=config["training"]["epochs"],
        model_dir=save_model_dir
    )

    
if __name__ == "__main__":
    train(
        config_file=CONFIG_FILE,
        tokenizer_path=TOKENIZER_PATH,
        data_dir=DATA_DIR,
        save_model_dir=SAVE_MODEL_DIR,
        model_path=MODEL_PATH
    )
