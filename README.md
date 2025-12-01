# GPT2 Language Model in PyTorch

This repository contains an implementation of the GPT2 language model in **pytorch**. It provides a full pipeline for training, generating, and evaluating language models with different tokenizers.

## Project Overview

- **Model**: Custom GPT2 implemented in PyTorch, supporting configurable architecture and hyperparameters.  
- **Tokenizers**: Supports any tokenizer in JSON format compatible with `tokenizers.Tokenizer`. Additionally, scripts are provided to build three types of tokenizers:
  - Whitespace tokenizer
  - BPE (Byte-Pair Encoding) tokenizer
  - SPLADE tokenizer (downloaded from the internet)  

- **Metrics**: The repository includes scripts for calculating evaluation metrics on trained models:
  - Word-level perplexity  
  - Character-level perplexity  
  - Out-of-Vocabulary (OOV) rate (dedicated for Whitespace tokenizer)
  - Tokenizer throughput (tokens/sec)  
  - Average tokens per word  
  - Number and percentage of words present directly in the tokenizer dictionary  

- **Configs**: Predefined configurations for models and training are stored in `configs/`. Training with a new setup requires adding a `.yaml` file in this folder.

## Requirements

- Python 3.13
- `uv`
- `PyTorch`
- `tokenizers` library
- Other dependencies listed in `pyproject.toml`

## Usage

```{bash}
uv sync
```

All scripts require adjusting variables at the top of the file (in CAPS) such as paths to data, models, or directories for logs and results.

### Build or Download Tokenizer

- Build a tokenizer on your own data or download SPLADE:  

```{bash}
python3 -m scripts.build_tokenizer
```

### Download Data

- Fetch texts from [Wolne Lektury](https://wolnelektury.pl) for selected authors:  

```{bash}
python3 -m scripts.dataset.download_wolnelektury
```

> Adjust author names according to the Wolne Lektury API.

### Train Model

- Train (or fine-tune) GPT2 with a selected tokenizer and configuration:  

```{bash}
python3 -m scripts.generation.train_model
```

> Specify the `.yaml` config file for architecture, hyperparameters, and training options.

### Generate Text

- Generate text from a trained model with a prompt:  

```{bash}
python3 -m scripts.generation.generate
```

### Calculate Metrics

- Compute evaluation metrics for a trained model and tokenizer on test data:  
```{bash}
python3 -m scripts.generation.calculate_metrics
```

## Project Structure

- `scripts/` – scripts for building tokenizers, downloading data, training models, generating text, and calculating metrics  
- `configs/` – YAML configuration files for models and training  
- `src/` - the source code for model
- `data/` – default location for dataset storage
- `models/` – default location for saved checkpoints of trained models

This README provides all instructions to prepare tokenizers, train models, generate text, and evaluate performance using the provided scripts.
