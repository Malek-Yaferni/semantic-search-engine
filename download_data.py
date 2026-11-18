
from datasets import load_dataset
import json
from pathlib import Path

# Project root: folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Create output folders
for folder in ["corpus", "queries", "qrels"]:
    (RAW_DIR / folder).mkdir(parents=True, exist_ok=True)


def get_first_split(dataset, name):
    """Get the first available split and display its name."""
    split_names = list(dataset.keys())
    print(f"{name} available splits: {split_names}")

    if not split_names:
        raise ValueError(f"No splits found for {name}")

    split_name = split_names[0]
    return dataset[split_name]


def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


print("Loading SciFact...")

corpus = load_dataset("BeIR/scifact", "corpus")
queries = load_dataset("BeIR/scifact", "queries")
qrels = load_dataset("BeIR/scifact-qrels")

print("\nExporting corpus...")
corpus_data = get_first_split(corpus, "Corpus")
save_jsonl(corpus_data, RAW_DIR / "corpus" / "corpus.jsonl")

print("\nExporting queries...")
queries_data = get_first_split(queries, "Queries")
save_jsonl(queries_data, RAW_DIR / "queries" / "queries.jsonl")

print("\nExporting qrels...")
qrels_count = 0

with open(RAW_DIR / "qrels" / "qrels.jsonl", "w", encoding="utf-8") as f:
    for split_name in qrels.keys():
        print(f"Processing qrels split: {split_name}")

        for item in qrels[split_name]:
            row = dict(item)
            row["split"] = split_name
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qrels_count += 1

print("\n========== EXPORT COMPLETED ==========")
print(f"Corpus documents: {len(corpus_data)}")
print(f"Queries: {len(queries_data)}")
print(f"Qrels judgments: {qrels_count}")
print(f"Saved in: {RAW_DIR}")
