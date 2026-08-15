"""
Fix for Phase 1 limitation #1: ai essays average ~3x the length of human/hybrid
essays, which would let a classifier "detect AI" on raw length alone.

Non-destructive fix: add a length-controlled text variant alongside the original.
Every essay's text field is left untouched; a new text_length_controlled field caps
each essay at WORD_CAP words, truncated at the last full sentence boundary at or
before the cap. Essays already under the cap are copied through unchanged (no padding).
"""
import json
import statistics
from pathlib import Path

from text_utils import split_sentences, word_count

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "essays.jsonl"

WORD_CAP = 1000


# Truncates at the last full sentence at or before WORD_CAP words - never mid-sentence,
# never padded. build_sentences.py's truncate_sentence_list() mirrors this exact
# algorithm (returning the sentence list instead of a rejoined string) to stay consistent.
def truncate_to_cap(text: str) -> str:
    if word_count(text) <= WORD_CAP:
        return text
    sentences = split_sentences(text)
    kept = []
    total = 0
    for s in sentences:
        w = word_count(s)
        if total + w > WORD_CAP and kept:
            break
        kept.append(s)
        total += w
    return " ".join(kept)


# Entry point: adds text_length_controlled/word_count_length_controlled to every record
# in-place and rewrites essays.jsonl, then prints a before/after mean-length summary.
def main():
    records = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    print("=" * 60)
    print("LENGTH NORMALIZATION (fix for limitation #1)")
    print("=" * 60)
    print(f"Word cap: {WORD_CAP}")

    for r in records:
        normalized = truncate_to_cap(r["text"])
        r["text_length_controlled"] = normalized
        r["word_count_length_controlled"] = word_count(normalized)

    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nBefore -> after (mean word count), by class:")
    for label in ("human", "ai", "hybrid"):
        before = [r["word_count"] for r in records if r["label"] == label]
        after = [r["word_count_length_controlled"] for r in records if r["label"] == label]
        print(f"  {label}: {statistics.mean(before):.0f} -> {statistics.mean(after):.0f} "
              f"(max after: {max(after)})")

    over_cap = [r["id"] for r in records if r["word_count_length_controlled"] > WORD_CAP]
    print(f"\nEssays still over cap after truncation: {len(over_cap)} (should be 0)")
    if over_cap:
        print(f"  {over_cap}")


if __name__ == "__main__":
    main()
