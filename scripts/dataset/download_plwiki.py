from datasets import load_dataset

DATASET_NAME = "chrisociepa/wikipedia-pl-20230401"
OUTPUT_DIR = "data/wiki_pl_raw_data"

print(f"Downloading dataset: {DATASET_NAME}...")

dataset = load_dataset(DATASET_NAME, split="train")
print(f"Initial number of articles: {len(dataset)}")

print(f"Saving dataset to: {OUTPUT_DIR}...")
dataset.save_to_disk(OUTPUT_DIR)

print(f"\nDataset saved to: {OUTPUT_DIR}")