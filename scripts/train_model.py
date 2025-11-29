import yaml
import torch
from torch.utils.data import DataLoader, random_split
from src.model.model_config import LanguageModelConfig
from src.model.transformer_model import LanguageModel
from src.training.trainer import Trainer
from src.data.text_dataset import TextDataset
from src.data.hf_dataset_processor import HFDatasetProcessor
from src.utils.model_utils import load_model
from tokenizers import Tokenizer

CONFIG_FILE = "configs/small-plwiki.yaml"
TOKENIZER_PATH = "tokenizers/polish-splade.json"
SAVE_MODEL_DIR = "models/model-plwiki"
PRETRAINED_MODEL_PATH = "models/model-plwiki/best"
VALIDATION_SET_PERC = 0.1

def train(config_file, tokenizer_path, save_model_dir, pretrained_model_path=None, starting_epoch=0, validation_set_perc=0.1):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Model
    if pretrained_model_path:
        model = load_model(pretrained_model_path)
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
    # dataset = TextDataset(
    #     data_dir=data_dir,
    #     tokenizer=tokenizer,
    #     seq_len=config["training"]["max_seq_len"]
    # )
    hf_dataset_processor = HFDatasetProcessor(
        data_path=config["data"]["dataset_path"],
        tokenizer=tokenizer,
        max_seq_len=config["training"]["max_seq_len"]
    )
    dataset = hf_dataset_processor.get_data()

    train_data, val_data = random_split(
        dataset=dataset, 
        lengths=[1 - validation_set_perc, validation_set_perc]
    )

    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8
    )

    val_loader = DataLoader(
        dataset=val_data, 
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8
    )

    # Training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"])
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    print("STARTED")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        model_dir=save_model_dir,
    )
    trainer.current_epoch = starting_epoch
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        save_shapshot_every=3
    )

    
if __name__ == "__main__":
    train(
        config_file=CONFIG_FILE,
        tokenizer_path=TOKENIZER_PATH,
        save_model_dir=SAVE_MODEL_DIR,
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        starting_epoch=1,
        validation_set_perc=VALIDATION_SET_PERC
    )
