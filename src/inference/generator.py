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

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = 50):
        self.model.eval()
        input_tokens = torch.tensor([self.tokenizer.encode(prompt).ids], device=self.device)

        with torch.no_grad():
            for k in range(max_new_tokens):
                if temperature == 0:
                    logits = self.model(input_tokens)[:, -1, :]
                else:
                    logits = self.model(input_tokens)[:, -1, :] / temperature

                probs = F.softmax(logits, dim=-1)

                if temperature == 0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    top_probs, top_idx = torch.topk(probs, k=top_k)
                    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                    sample_idx = torch.multinomial(top_probs, 1)
                    next_token = top_idx.gather(-1, sample_idx)

                input_tokens = torch.cat([input_tokens, next_token], dim=1)

                if next_token.item() == self.eos_token_id:
                    break   

        return self.tokenizer.decode(input_tokens[0].tolist())
    
    def batch_generate(self, prompts: str, **kwargs):
        results = [self.generate(prompt,  **kwargs) for prompt in prompts]
        return results