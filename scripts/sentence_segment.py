"""Phase 1, Step 6: fill in sentence_count for every essay in dataset/essays.jsonl."""
import json
from pathlib import Path

from text_utils import split_sentences, using_spacy

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "essays.jsonl"


def main():
    records = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    print("=" * 60)
    print("STEP 6: SENTENCE SEGMENTATION")
    print("=" * 60)
    print(f"Splitter: {'spaCy (en_core_web_sm)' if using_spacy() else 'regex fallback'}")

    for r in records:
        r["sentence_count"] = len(split_sentences(r["text"]))

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = [r["sentence_count"] for r in records]
    print(f"Filled sentence_count for {len(records)} essays.")
    print(f"Range: {min(counts)}-{max(counts)} sentences, avg {sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    main()
