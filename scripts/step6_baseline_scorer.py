"""
Phase 2, Step 6: weighted-sum baseline scorer over the (clipped) z-scored features.
Explicitly a sanity-check reference point for Step 7's logistic regression, not the
final model. Uses only the 4 features with an established, matching direction
(perplexity, logprob_variance, sentence_length_variance, syntactic_depth_variance);
features 5-7 (perplexity_variance, ngram_repeat_rate, transition_repetition_rate) are
excluded from this simple baseline because they don't have a trustworthy established
direction to sign a weight by - Step 7's trained model still sees all 7 and decides
their fate with real coefficients, this is just a simpler reference point.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
ZSCORED_PATH = ROOT / "dataset" / "features_train_zscored.jsonl"
CLIPPED_PATH = ROOT / "dataset" / "features_train_clipped.jsonl"
SCORES_PATH = ROOT / "dataset" / "baseline_scores_train.jsonl"

ALL_Z_FIELDS = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]

# Step 6b weights: features 1-4 only. Every one of these established "ai LOWER" as the
# matching direction (Steps 4-5), so all four get an equal-magnitude weight of 1 with a
# negative sign - a lower z-score (more AI-like) should contribute POSITIVELY to the
# AI-likelihood score, so negating a low (negative) z-score produces a positive
# contribution. No feature is weighted higher than another: this is a baseline sanity
# check, not a tuned model, so unequal weights would just be guessing.
WEIGHTS = {
    "perplexity_z": -1.0,               # ai LOWER perplexity -> negate so low z -> +score
    "logprob_variance_z": -1.0,         # ai LOWER logprob variance -> same reasoning
    "sentence_length_variance_z": -1.0, # ai LOWER burstiness (sentence length) -> same
    "syntactic_depth_variance_z": -1.0, # ai LOWER burstiness (syntactic depth) -> same
}
CLIP_LO, CLIP_HI = -5.0, 5.0


def main():
    rows = [json.loads(l) for l in ZSCORED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 6a: Z-SCORE CLIPPING [-5, 5]")
    print("=" * 70)

    clip_counts = {f: 0 for f in ALL_Z_FIELDS}
    for r in rows:
        for f in ALL_Z_FIELDS:
            v = r.get(f)
            if v is None:
                continue
            if v < CLIP_LO:
                r[f] = CLIP_LO
                clip_counts[f] += 1
            elif v > CLIP_HI:
                r[f] = CLIP_HI
                clip_counts[f] += 1

    total_clipped = sum(clip_counts.values())
    print(f"Total values clipped: {total_clipped}")
    for f, c in clip_counts.items():
        print(f"  {f}: {c}")

    with CLIPPED_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} rows to {CLIPPED_PATH.relative_to(ROOT)} "
          f"(authoritative going forward - features_train_zscored.jsonl is left as the "
          f"unclipped record).")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6b: WEIGHTS")
    print("=" * 70)
    for f, w in WEIGHTS.items():
        print(f"  {f}: weight = {w:+.1f}  (established direction: ai LOWER -> negate)")
    print("  perplexity_variance_z, ngram_repeat_rate_z, transition_repetition_rate_z: "
          "EXCLUDED (no trustworthy established direction for this simple baseline)")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6c: COMPUTE WEIGHTED-SUM SCORE")
    print("=" * 70)

    # Verify essay-level joins: sentence_length_variance_z / syntactic_depth_variance_z
    # must be constant across every sentence row of a given essay.
    essay_vals = {}
    join_mismatches = []
    for r in rows:
        eid = r["essay_id"]
        key = (r.get("sentence_length_variance_z"), r.get("syntactic_depth_variance_z"))
        if eid not in essay_vals:
            essay_vals[eid] = key
        elif essay_vals[eid] != key:
            join_mismatches.append(eid)
    if join_mismatches:
        print(f"!! FLAGGED: essay-level join mismatch in {len(set(join_mismatches))} essay(s): "
              f"{sorted(set(join_mismatches))[:10]}")
    else:
        print(f"Essay-level join check: OK - sentence_length_variance_z and "
              f"syntactic_depth_variance_z are constant across all sentence rows for all "
              f"{len(essay_vals)} essays (post segmentation-fix values, confirmed consistent).")

    score_rows = []
    skipped = 0
    for r in rows:
        if any(r.get(f) is None for f in WEIGHTS):
            # logprob_variance_z can be None (Step 2's token_logprob_variance() returns
            # None for sentences with <2 scorable tokens - e.g. a 1-token sentence with
            # no preceding context). Can't compute a 4-term sum with a missing term, so
            # these rows are skipped from the baseline score rather than silently
            # treating None as 0 (which would misrepresent them as perfectly average).
            skipped += 1
            continue
        score = sum(WEIGHTS[f] * r[f] for f in WEIGHTS)
        score_rows.append({
            "essay_id": r["essay_id"],
            "sentence_idx": r["sentence_idx"],
            "sentence_label": r["sentence_label"],
            "score": score,
        })

    print(f"\nSkipped {skipped} rows with a missing feature value (logprob_variance_z is "
          f"None for sentences with <2 scorable tokens) - not scored, not silently "
          f"imputed as 0.")

    with SCORES_PATH.open("w", encoding="utf-8") as f:
        for r in score_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(score_rows)} rows to {SCORES_PATH.relative_to(ROOT)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6d: EVALUATE SEPARATION")
    print("=" * 70)

    by_label = {"human": [], "ai": [], "ai_edited": []}
    for r in score_rows:
        by_label[r["sentence_label"]].append(r["score"])

    print("\nScore distribution by sentence_label:")
    means = {}
    for label, vals in by_label.items():
        m, s = statistics.mean(vals), statistics.pstdev(vals)
        means[label] = m
        print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={len(vals)}")

    ai_all = by_label["ai"] + by_label["ai_edited"]
    ai_mean = statistics.mean(ai_all)
    human_mean = means["human"]
    threshold = (human_mean + ai_mean) / 2
    print(f"\nThreshold (midpoint of human mean {human_mean:.4f} and ai/ai_edited mean "
          f"{ai_mean:.4f}): {threshold:.4f}")

    tp = sum(1 for v in ai_all if v > threshold)
    fn = sum(1 for v in ai_all if v <= threshold)
    tn = sum(1 for v in by_label["human"] if v <= threshold)
    fp = sum(1 for v in by_label["human"] if v > threshold)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    print(f"\nAt threshold {threshold:.4f} (positive class = ai/ai_edited):")
    print(f"  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
    print(f"  Accuracy  = {accuracy:.4f}")
    print(f"  Precision = {precision:.4f}")
    print(f"  Recall    = {recall:.4f}")

    print(f"\nDoes the 4-feature weighted sum separate the classes reasonably well? "
          f"{'YES' if accuracy > 0.65 else 'PARTIALLY' if accuracy > 0.55 else 'NO'} "
          f"(accuracy {accuracy:.1%} vs. a 50% no-signal baseline).")


if __name__ == "__main__":
    main()
