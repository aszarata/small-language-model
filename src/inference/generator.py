import torch
import torch.nn.functional as F

class Generator:
    def __init__(self, model, tokenizer, eos_token="</s>"):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = model
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.eos_token_id = tokenizer.token_to_id(eos_token)

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0):
        self.model.eval()
        input_tokens = torch.tensor([self.tokenizer.encode(prompt).ids], device=self.device)

        with torch.no_grad():
            for k in range(max_new_tokens):
                logits = self.model(input_tokens)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_tokens = torch.cat([input_tokens, next_token], dim=1)

                if next_token.item() == self.eos_token_id:
                    break   

        return self.tokenizer.decode(input_tokens[0].tolist())