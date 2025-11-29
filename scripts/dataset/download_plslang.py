from datasets import load_dataset

DATASET_NAME = "jziebura/polish_youth_slang_classification"
OUTPUT_DIR = "data/slang_pl_raw_data"

print(f"Downloading dataset: {DATASET_NAME}...")

dataset = load_dataset(DATASET_NAME, split="train")
print(f"Initial number of articles: {len(dataset)}")

print(f"Saving dataset to: {OUTPUT_DIR}...")
dataset.save_to_disk(OUTPUT_DIR)

print(f"\nDataset saved to: {OUTPUT_DIR}")