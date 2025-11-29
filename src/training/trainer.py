import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, criterion, tokenizer, device=None, model_dir=None, classification=False):
        if not device:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = device
        self.model = model
        model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.current_epoch = 0
        self.classification = classification

        self.logger = setup_logger(name="Trainer", log_dir=f"{model_dir}/logs/training")
        self.model_dir = model_dir

        self.train_losses = []
        self.val_losses = []

        self.best_val_loss = torch.inf

        self.save_tokenizer()

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                x, y = batch['input_ids'], batch['labels']
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(x)
                if not self.classification:
                    loss = self.criterion(
                        out.view(-1, out.size(-1)), 
                        y.view(-1)
                    )
                else:
                    loss = self.criterion(out, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                x, y = batch['input_ids'], batch['labels']
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                if not self.classification:
                    loss = self.criterion(
                        out.view(-1, out.size(-1)), 
                        y.view(-1)
                    )
                else:
                    loss = self.criterion(out, y)
                total_loss += loss.item()

        val_loss = total_loss / len(dataloader)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_snapshot(name="best")
            
        return val_loss

    def fit(self, train_loader, val_loader=None, epochs=10, save_shapshot_every=5):
        start_epoch = self.current_epoch + 1
        end_epoch = self.current_epoch + epochs + 1

        self.logger.info(f"Training started for epochs: {start_epoch} - {end_epoch-1}")
        self.logger.info(f"{self.model.config}") 
        self.logger.info(f"Using device {self.device}")
    
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader) if val_loader else None

            self.train_losses.append(train_loss)
            if val_loader is not None: self.val_losses.append(val_loss)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}" + (f", val_loss={val_loss:.4f}" if val_loss else ""))

            if (self.current_epoch) % save_shapshot_every == 0:
                self.save_snapshot(name=f"checkpoint_{self.current_epoch}")
            
            self.save_snapshot()
            self.save_loss_plot()

    def save_snapshot(self, name="last"):
        if not self.model_dir:
            return
        output_dir = f"{self.model_dir}/{name}"
        os.makedirs(output_dir, exist_ok=True)

        self.model.config.save(f"{output_dir}/config.json")
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{output_dir}/optimizer.pt")

    def save_tokenizer(self):
        if not self.model_dir:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        self.tokenizer.save(f"{self.model_dir}/tokenizer.json")

    def save_loss_plot(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(f'{self.model_dir}/loss_plot.png')
        plt.close()
