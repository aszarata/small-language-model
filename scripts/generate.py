from tokenizers import Tokenizer
from src.utils.model_utils import load_model
from src.inference.generator import Generator

MODEL_DIR = "models/model-bpe/checkpoint_6"
TOKENIZER_PATH = "models/model-bpe/tokenizer.json"
PROMPT = """Był użyty jako dekoracja, jako wspaniała brama architektoniczna, wiodąca z życia do śmierci. Grał rolę metafory w tym obrzędzie, który odbywał się tu przez długi czas z nieodmiennym codziennym ceremoniałem. Ludzie, zmęczeni drogą, jeszcze żywi, jeszcze będący sobą, we własnych ubraniach podróżnych, mijali jedną i drugą bramę i wjeżdżali na wewnętrzny dziedziniec rezydencji. Z ciężarowego samochodu odpadały tylne drzwi, podróżni, pomagając sobie nawzajem, wstępowali tłumnie po stopniach schodów, mogąc jeszcze mniemać, że — według napisu nad wejściem — wchodzą do Zakładu Kąpielowego. Po pewnym czasie, przebywszy w poprzek wnętrze gmachu, ukazywali się na ganku po jego stronie przeciwnej już tylko w bieliźnie — niektórzy jeszcze z kawałkiem mydła i ręcznikiem w dłoni. Przynaglani do pośpiechu, uchylając się od uderzeń kolbami, wbiegali"""
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