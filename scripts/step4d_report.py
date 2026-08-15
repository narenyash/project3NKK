"""
Phase 2, Step 4d: full-scale descriptive report over dataset/features_train_raw.jsonl.
Reports row counts, mean/std per feature per sentence_label, all 7 directional checks at
full scale, and an anomaly scan for any new h_066-style outlier essays.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "dataset" / "features_train_raw.jsonl"

SENTENCE_FIELDS = ["perplexity", "logprob_variance"]
ESSAY_FIELDS = ["sentence_length_variance", "syntactic_depth_variance", "perplexity_variance",
                "bigram_repeat_rate", "trigram_repeat_rate", "ngram_repeat_rate",
                "transition_repetition_rate"]


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    rows = [json.loads(l) for l in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    print("=" * 70)
    print("STEP 4d: FULL-SCALE DESCRIPTIVE REPORT (train split)")
    print("=" * 70)

    from collections import Counter
    by_label = Counter(r["sentence_label"] for r in rows)
    print(f"\nTotal sentence rows: {len(rows)}")
    print("Row counts by sentence_label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")

    print("\n--- Sentence-level features: mean (std) by sentence_label ---")
    for field in SENTENCE_FIELDS:
        print(f"\n{field}:")
        for label in ("human", "ai", "ai_edited"):
            vals = [r[field] for r in rows if r["sentence_label"] == label]
            m, s = mean_std(vals)
            print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={len([v for v in vals if v is not None])}")

    # Essay-level features: dedupe to one row per essay per essay_label
    seen = {}
    for r in rows:
        if r["essay_id"] not in seen:
            seen[r["essay_id"]] = r
    essay_rows = list(seen.values())
    by_essay_label = Counter(r["essay_label"] for r in essay_rows)
    print(f"\nEssay-level dedup: {len(essay_rows)} essays "
          f"(human={by_essay_label['human']}, ai={by_essay_label['ai']}, hybrid={by_essay_label['hybrid']})")

    print("\n--- Essay-level features: mean (std) by essay_label ---")
    for field in ESSAY_FIELDS:
        print(f"\n{field}:")
        for label in ("human", "ai", "hybrid"):
            vals = [r[field] for r in essay_rows if r["essay_label"] == label]
            m, s = mean_std(vals)
            print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={len([v for v in vals if v is not None])}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("7 DIRECTIONAL CHECKS AT FULL SCALE")
    print("=" * 70)

    human_sent = [r for r in rows if r["sentence_label"] == "human"]
    ai_sent = [r for r in rows if r["sentence_label"] in ("ai", "ai_edited")]

    def sm(rows_, field):
        vals = [r[field] for r in rows_ if r[field] is not None]
        return statistics.mean(vals) if vals else None

    def em(label, field):
        vals = [r[field] for r in essay_rows if r["essay_label"] == label and r[field] is not None]
        return statistics.mean(vals) if vals else None

    checks = []

    ppl_h, ppl_a = sm(human_sent, "perplexity"), sm(ai_sent, "perplexity")
    checks.append(("1. perplexity", ppl_h, ppl_a, ppl_a < ppl_h, "ai LOWER"))

    var_h, var_a = sm(human_sent, "logprob_variance"), sm(ai_sent, "logprob_variance")
    checks.append(("2. logprob_variance", var_h, var_a, var_a < var_h, "ai LOWER"))

    slv_h, slv_a = em("human", "sentence_length_variance"), em("ai", "sentence_length_variance")
    checks.append(("3. sentence_length_variance", slv_h, slv_a, slv_a < slv_h, "ai LOWER"))

    sdv_h, sdv_a = em("human", "syntactic_depth_variance"), em("ai", "syntactic_depth_variance")
    checks.append(("4. syntactic_depth_variance", sdv_h, sdv_a, sdv_a < sdv_h, "ai LOWER"))

    pv_h, pv_a = em("human", "perplexity_variance"), em("ai", "perplexity_variance")
    checks.append(("5. perplexity_variance", pv_h, pv_a, pv_a < pv_h, "ai LOWER"))

    ngr_h, ngr_a = em("human", "ngram_repeat_rate"), em("ai", "ngram_repeat_rate")
    checks.append(("6. ngram_repeat_rate", ngr_h, ngr_a, ngr_a > ngr_h, "ai HIGHER"))

    tr_h, tr_a = em("human", "transition_repetition_rate"), em("ai", "transition_repetition_rate")
    checks.append(("7. transition_repetition_rate", tr_h, tr_a, tr_a > tr_h, "ai HIGHER"))

    n_match = 0
    for name, h_val, a_val, matches, expect in checks:
        n_match += matches
        print(f"\n{name}:")
        print(f"  human = {h_val:.4f}, ai/ai_edited = {a_val:.4f}")
        print(f"  Expected: {expect}. {'MATCH' if matches else 'NO MATCH'}")

    print(f"\n{n_match}/7 checks matched the expected direction at full scale.")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANOMALY SCAN: any new h_066-style outlier essays?")
    print("=" * 70)
    pv_vals = sorted(
        [(r["essay_id"], r["essay_label"], r["perplexity_variance"]) for r in essay_rows if r["perplexity_variance"] is not None],
        key=lambda x: -x[2]
    )
    print("\nTop 10 perplexity_variance essays (highest first):")
    for eid, label, v in pv_vals[:10]:
        print(f"  {eid} ({label}): {v:,.1f}")

    ppl_by_essay = {}
    for r in rows:
        ppl_by_essay.setdefault(r["essay_id"], []).append(r["perplexity"])
    max_ppl_per_essay = sorted(
        [(eid, max(v for v in vals if v is not None)) for eid, vals in ppl_by_essay.items() if any(v is not None for v in vals)],
        key=lambda x: -x[1]
    )
    print("\nTop 10 essays by single-highest sentence perplexity (possible remaining junk fragments):")
    for eid, v in max_ppl_per_essay[:10]:
        print(f"  {eid}: max sentence perplexity = {v:,.1f}")


if __name__ == "__main__":
    main()
