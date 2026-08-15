"""
Phase 2, Step 4c: full-scale feature extraction over the entire train split of the
(now junk-fragment-filtered) sentences.jsonl. Reuses the same single-forward-pass logic
from Step 2/3 for perplexity + log-prob variance. All four functions run unmodified,
including the three flagged in Step 3/4b (perplexity_variance, ngram_repeat_rate,
transition_repetition_rate) - no changes made to any of them.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F

ROOT = Path(__file__).resolve().parent.parent
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"
OUT_PATH = ROOT / "dataset" / "features_train_raw.jsonl"


def main():
    sentences = [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    train_sentences = [r for r in sentences if r["split"] == "train"]

    rows_by_essay = {}
    for r in train_sentences:
        rows_by_essay.setdefault(r["essay_id"], []).append(r)
    for eid in rows_by_essay:
        rows_by_essay[eid].sort(key=lambda r: r["sentence_idx"])

    print("=" * 70)
    print("STEP 4c: FULL-SCALE EXTRACTION (train split)")
    print("=" * 70)
    print(f"{len(train_sentences)} sentences across {len(rows_by_essay)} essays")
    print("Loading GPT-2...")
    F.get_model()

    feature_rows = []
    essay_ids = sorted(rows_by_essay.keys())
    for i, eid in enumerate(essay_ids, 1):
        essay_rows = rows_by_essay[eid]
        sentences_text = [r["sentence_text"] for r in essay_rows]

        perplexities = []
        logprob_vars = []
        for idx, s in enumerate(sentences_text):
            start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
            context = " ".join(sentences_text[start:idx])
            token_lps = F.sentence_token_logprobs(s, context)
            perplexities.append(F.sentence_perplexity(token_lps))
            logprob_vars.append(F.token_logprob_variance(token_lps))

        burst = F.essay_burstiness(sentences_text, perplexities)
        ngram = F.essay_ngram_repetition(sentences_text)

        for idx, r in enumerate(essay_rows):
            feature_rows.append({
                "essay_id": eid,
                "sentence_idx": r["sentence_idx"],
                "sentence_label": r["sentence_label"],
                "essay_label": r["essay_label"],
                "topic": r["topic"],
                "perplexity": perplexities[idx],
                "logprob_variance": logprob_vars[idx],
                "sentence_length_variance": burst["sentence_length_variance"],
                "syntactic_depth_variance": burst["syntactic_depth_variance"],
                "perplexity_variance": burst["perplexity_variance"],
                "bigram_repeat_rate": ngram["bigram_repeat_rate"],
                "trigram_repeat_rate": ngram["trigram_repeat_rate"],
                "ngram_repeat_rate": ngram["ngram_repeat_rate"],
                "transition_repetition_rate": ngram["transition_repetition_rate"],
            })

        if i % 25 == 0 or i == len(essay_ids):
            print(f"[{i}/{len(essay_ids)}] essays done ({len(feature_rows)} sentence rows so far)", flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in feature_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(feature_rows)} feature rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
