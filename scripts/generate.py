from tokenizers import Tokenizer
from src.utils.model_utils import load_model
from src.inference.generator import Generator

MODEL_DIR = "models/model-bpe/checkpoint_6"
TOKENIZER_PATH = "models/model-bpe/tokenizer.json"
PROMPT = """Pałac, którego już nie ma stoi na"""
MAX_NEW_TOKENS = 32
TEMPERATURE = 0


def generate(prompt, model_dir, tokenizer_path, max_new_tokens, temperature):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = load_model(model_dir)

    generator = Generator(
        model=model,
        tokenizer=tokenizer
    )

    reply = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    print(reply)

if __name__ == "__main__":
    generate(
        prompt=PROMPT,
        model_dir=MODEL_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )