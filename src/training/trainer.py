import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, tokenizer, device=None):
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

        self.model_dir = None

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        with tqdm(dataloader, desc="Training") as pbar:
            for (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(x)
                loss = self.criterion(
                    out.view(-1, out.size(-1)), 
                    y.view(-1)
                )

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
            for (x, y) in tqdm(dataloader, desc="Validation"):
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    out = self.model(x)
                    loss = self.criterion(
                        out.view(-1, out.size(-1)), 
                        y.view(-1)
                    )
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader=None, epochs=10, model_dir=None):
        self.model_dir = model_dir
        self.save_tokenizer()
        print(f"Starting training for {epochs} epochs. Using device {self.device}")
        
        start_epoch = self.current_epoch + 1
        end_epoch = self.current_epoch + epochs + 1
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader) if val_loader else None
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}" + (f", val_loss={val_loss:.4f}" if val_loss else ""))
            self.save_snapshot()

    def save_snapshot(self):
        if not self.model_dir:
            return
        output_dir = f"{self.model_dir}/checkpoint_{self.current_epoch}"
        os.makedirs(output_dir, exist_ok=True)

        self.model.config.save(f"{output_dir}/config.json")
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{output_dir}/optimizer.pt")

    def save_tokenizer(self):
        if not self.model_dir:
            return
        os.makedirs(self.model_dir, exist_ok=True)
        self.tokenizer.save(f"{self.model_dir}/tokenizer.json")
