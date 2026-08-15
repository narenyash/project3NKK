"""
Phase 2, Step 2: spot-test the four feature functions, one at a time, on small
hand-picked samples. No full-scale run over sentences.jsonl happens here.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F

ROOT = Path(__file__).resolve().parent.parent
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"


def load_sentences():
    return [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def context_for(rows_by_essay, essay_id, idx):
    rows = rows_by_essay[essay_id]
    start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
    return " ".join(r["sentence_text"] for r in rows[start:idx])


def main():
    rows = load_sentences()
    rows_by_essay = {}
    for r in rows:
        rows_by_essay.setdefault(r["essay_id"], []).append(r)
    for eid in rows_by_essay:
        rows_by_essay[eid].sort(key=lambda r: r["sentence_idx"])

    print("Loading GPT-2...")
    F.get_model()

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FUNCTIONS 1 & 2: sentence_perplexity() + token_logprob_variance()")
    print("(single shared forward pass per sentence)")
    print("=" * 70)

    picks = [
        ("h_001", 2, "human, generic-ish opening line"),
        ("ai_001", 0, "ai, generic opening line"),
        ("h_006", 3, "human, distinctive/idiosyncratic (rock collection)"),
        ("hy_050", 7, "hybrid ai_edited, distinctive/flowery claim"),
        ("h_016", 3, "human, distinctive (kombucha CO2 detail)"),
    ]
    for eid, idx, note in picks:
        rows_for_essay = rows_by_essay[eid]
        if idx >= len(rows_for_essay):
            print(f"  (skipping {eid}[{idx}] - out of range)")
            continue
        row = rows_for_essay[idx]
        context = context_for(rows_by_essay, eid, idx)
        token_lps = F.sentence_token_logprobs(row["sentence_text"], context)
        ppl = F.sentence_perplexity(token_lps)
        var = F.token_logprob_variance(token_lps)
        print(f"\n[{eid}#{idx}] ({row['sentence_label']}) {note}")
        print(f"  text: {row['sentence_text'][:100]!r}")
        print(f"  perplexity = {ppl}")
        print(f"  logprob_variance = {var}")
        print(f"  n_scored_tokens = {len(token_lps)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FUNCTION 3: essay_burstiness()")
    print("=" * 70)

    for eid, label in [("h_006", "human"), ("ai_001", "ai")]:
        essay_rows = rows_by_essay[eid]
        sentences = [r["sentence_text"] for r in essay_rows]
        perplexities = []
        for idx, s in enumerate(sentences):
            context = context_for(rows_by_essay, eid, idx)
            token_lps = F.sentence_token_logprobs(s, context)
            perplexities.append(F.sentence_perplexity(token_lps))
        burst = F.essay_burstiness(sentences, perplexities)
        print(f"\n[{eid}] ({label}, {len(sentences)} sentences)")
        for k, v in burst.items():
            print(f"  {k} = {v:.4f}" if v is not None else f"  {k} = None")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FUNCTION 4: essay_ngram_repetition() (whole-essay, redesigned v4)")
    print(f"Transition word list ({len(F.TRANSITION_WORDS)}): {F.TRANSITION_WORDS}")
    print("=" * 70)

    for eid, label in [("h_006", "human"), ("ai_001", "ai"), ("h_016", "human"), ("hy_050", "hybrid")]:
        sentences = [r["sentence_text"] for r in rows_by_essay[eid]]
        result = F.essay_ngram_repetition(sentences)
        print(f"\n[{eid}] ({label}, {len(sentences)} sentences)")
        for k, v in result.items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
