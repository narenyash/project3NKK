"""
Phase 2, Step 7.5b: full-scale extraction of the 3 new local features over the entire
train split, joined onto the existing feature table.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "dataset" / "features_train_raw.jsonl"
OUT_PATH = ROOT / "dataset" / "features_train_with_local.jsonl"


def main():
    rows = [json.loads(l) for l in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_essay = {}
    for r in rows:
        by_essay.setdefault(r["essay_id"], []).append(r)
    for eid in by_essay:
        by_essay[eid].sort(key=lambda r: r["sentence_idx"])

    print("=" * 70)
    print("STEP 7.5b: FULL-SCALE LOCAL FEATURE EXTRACTION (train split)")
    print("=" * 70)
    print(f"{len(rows)} sentences across {len(by_essay)} essays")

    essay_ids = sorted(by_essay.keys())
    for i, eid in enumerate(essay_ids, 1):
        essay_rows = by_essay[eid]
        sentences_text = None  # not needed directly; use sentence_text from raw rows? not present here
        perplexities = [r["perplexity"] for r in essay_rows]

        # Need sentence texts for length/depth - not stored in features_train_raw.jsonl,
        # so pull from sentences.jsonl for this essay.
        essay_rows_for_texts = SENTENCES_BY_ESSAY.get(eid)
        sentences_text = [r["sentence_text"] for r in essay_rows_for_texts]

        local_ppl_dev = [F.local_perplexity_deviation(perplexities, j) for j in range(len(sentences_text))]
        burst = F.local_burstiness_disruption(sentences_text)

        for j, r in enumerate(essay_rows):
            r["local_perplexity_deviation"] = local_ppl_dev[j]
            r["local_length_deviation"] = burst["local_length_deviation"][j]
            r["local_depth_deviation"] = burst["local_depth_deviation"][j]

        if i % 25 == 0 or i == len(essay_ids):
            print(f"[{i}/{len(essay_ids)}] essays done", flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} rows to {OUT_PATH.relative_to(ROOT)}")

    print("\n--- Descriptive stats by sentence_label ---")
    for field in ["local_perplexity_deviation", "local_length_deviation", "local_depth_deviation"]:
        print(f"\n{field}:")
        for label in ("human", "ai", "ai_edited"):
            vals = [r[field] for r in rows if r["sentence_label"] == label and r.get(field) is not None]
            if not vals:
                print(f"  {label:10s} n=0")
                continue
            m, s = statistics.mean(vals), statistics.pstdev(vals)
            print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={len(vals)}")

    print("\n--- The actual test: within-hybrid-essay ai_edited vs human comparison ---")
    for field in ["local_perplexity_deviation", "local_length_deviation", "local_depth_deviation"]:
        human_in_hybrid = [abs(r[field]) for r in rows if r["essay_label"] == "hybrid" and r["sentence_label"] == "human" and r.get(field) is not None]
        ai_edited = [abs(r[field]) for r in rows if r["sentence_label"] == "ai_edited" and r.get(field) is not None]
        hm = statistics.mean(human_in_hybrid) if human_in_hybrid else None
        am = statistics.mean(ai_edited) if ai_edited else None
        print(f"{field}: human-within-hybrid |mean|={hm:.4f} (n={len(human_in_hybrid)}), "
              f"ai_edited |mean|={am:.4f} (n={len(ai_edited)})")


if __name__ == "__main__":
    SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"
    sentences_rows = [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    SENTENCES_BY_ESSAY = {}
    for r in sentences_rows:
        SENTENCES_BY_ESSAY.setdefault(r["essay_id"], []).append(r)
    for eid in SENTENCES_BY_ESSAY:
        SENTENCES_BY_ESSAY[eid].sort(key=lambda r: r["sentence_idx"])
    main()
