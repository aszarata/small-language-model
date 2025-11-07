import json
import requests
from pathlib import Path
from tokenizers import Tokenizer

def download_pretrained_tokenizer(output_dir, output_name, tokenizer_url):
    response = requests.get(tokenizer_url)
    
    if response.status_code == 200:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f"{output_dir}/{output_name}", "w") as f:
            json.dump(response.json(), f)
    
    else:
        raise Exception(f"Error while processing request: {response.status_code}")
    
    try:
        tokenizer = Tokenizer.from_file(f"{output_dir}/{output_name}")
    except Exception as e:
        Exception(f"Invalid tokenizer.json formatting. Exception: {e}")