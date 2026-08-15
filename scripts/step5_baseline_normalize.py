"""
Phase 2, Step 5: compute the human corpus baseline and z-score normalize all features.

Baseline scope: single GLOBAL human baseline (not per-topic), an explicit consequence of
Phase 1's finding that topic tagging is unreliable at the distribution level (94% of
essays landed in one topic category under 3-model consensus) - a per-topic split would
leave most topic buckets too small (many had 0-3 essays total) to produce a stable
baseline.

Sentence-level vs essay-level features: perplexity and logprob_variance are genuinely
per-sentence, so the baseline uses every human sentence row directly. The other five
features (sentence_length_variance, syntactic_depth_variance, perplexity_variance,
ngram_repeat_rate, transition_repetition_rate) are essay-level values joined onto every
sentence row for that essay (same convention used since Step 2/3/4) - computing their
baseline over raw sentence rows would over-weight essays with more sentences (since their
single essay-level value gets counted once per sentence). These are deduplicated to one
value per essay before computing mean/std, matching how Step 4's descriptive report was
already computed.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "dataset" / "features_train_raw.jsonl"
BASELINE_PATH = ROOT / "dataset" / "corpus_baseline.json"
ZSCORED_PATH = ROOT / "dataset" / "features_train_zscored.jsonl"

SENTENCE_LEVEL_FEATURES = ["perplexity", "logprob_variance"]
ESSAY_LEVEL_FEATURES = [
    "sentence_length_variance", "syntactic_depth_variance", "perplexity_variance",
    "ngram_repeat_rate", "transition_repetition_rate",
]
ALL_FEATURES = SENTENCE_LEVEL_FEATURES + ESSAY_LEVEL_FEATURES
FLAGGED_FEATURES = {"perplexity_variance", "ngram_repeat_rate", "transition_repetition_rate"}


def main():
    rows = [json.loads(l) for l in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    human_rows = [r for r in rows if r["sentence_label"] == "human"]

    print("=" * 70)
    print("STEP 5a: COMPUTE HUMAN BASELINE")
    print("=" * 70)
    print("Baseline scope: single GLOBAL human baseline, not per-topic - Phase 1's topic")
    print("tagging is unreliable at the distribution level (94% of essays landed in one")
    print("topic under 3-model consensus), so per-topic buckets would mostly be too small")
    print("(many topics had 0-3 essays total) to produce a stable baseline.\n")

    # IMPORTANT: essay-level features must come from essay_label == "human" essays only,
    # NOT from "any essay containing a human-labeled sentence." Hybrid essays contain
    # human-labeled sentences too (their verbatim-retained portions), so filtering by
    # sentence_label alone and deduplicating by essay_id would silently pull in all 78
    # hybrid essays' whole-essay structure alongside the 78 genuine human essays - a real
    # bug caught on the first run of this script (produced 156 "human essays" instead of
    # 78, i.e. 78 human + 78 hybrid). Sentence-level features (perplexity, logprob_
    # variance) are correctly scoped by sentence_label alone - a verbatim human sentence
    # embedded in a hybrid essay is still legitimately human-written at the sentence level.
    human_essay_rows = {}
    for r in rows:
        if r["essay_label"] == "human":
            human_essay_rows.setdefault(r["essay_id"], r)
    n_human_essays = len(human_essay_rows)

    print(f"Human sentence rows (sentence_label == 'human', any essay type): {len(human_rows)}")
    print(f"Human essays (essay_label == 'human', deduplicated): {n_human_essays}")
    print("Cross-check against Step 4's full-scale report: expected 78 human essays.\n")

    baseline = {}
    near_zero_std = []

    for field in SENTENCE_LEVEL_FEATURES:
        vals = [r[field] for r in human_rows if r[field] is not None]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        baseline[field] = {"mean": mean, "std": std}
        print(f"{field} (sentence-level, n={len(vals)}): mean={mean:.4f}, std={std:.4f}")
        if std < 1e-6:
            near_zero_std.append(field)

    for field in ESSAY_LEVEL_FEATURES:
        vals = [r[field] for r in human_essay_rows.values() if r[field] is not None]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        baseline[field] = {"mean": mean, "std": std}
        print(f"{field} (essay-level, n={len(vals)}): mean={mean:.4f}, std={std:.4f}")
        if std < 1e-6:
            near_zero_std.append(field)

    if near_zero_std:
        print(f"\n!! FLAGGED: near-zero std for {near_zero_std} - z-scores for these would be "
              "unstable. Holding off normalizing these specific features until reviewed.")
    else:
        print("\nNo near-zero-variance features found. All 7 safe to normalize.")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5b: PERSIST BASELINE")
    print("=" * 70)
    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"Wrote {BASELINE_PATH.relative_to(ROOT)}")
    print(json.dumps(baseline, indent=2))

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5c: APPLY Z-SCORE NORMALIZATION TO FULL TRAINING TABLE")
    print("=" * 70)

    skip_fields = set(near_zero_std)
    zscored_rows = []
    for r in rows:
        out = dict(r)  # keep all raw fields
        for field in ALL_FEATURES:
            if field in skip_fields:
                continue
            val = r[field]
            m, s = baseline[field]["mean"], baseline[field]["std"]
            out[f"{field}_z"] = (val - m) / s if (val is not None and s > 0) else None
        zscored_rows.append(out)

    with ZSCORED_PATH.open("w", encoding="utf-8") as f:
        for r in zscored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(zscored_rows)} rows to {ZSCORED_PATH.relative_to(ROOT)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5d: POST-NORMALIZATION SANITY REPORT")
    print("=" * 70)

    def zmean_std(rows_, field_z, dedupe_essay=False):
        if dedupe_essay:
            seen = {}
            for r in rows_:
                seen.setdefault(r["essay_id"], r)
            rows_ = list(seen.values())
        vals = [r[field_z] for r in rows_ if r.get(field_z) is not None]
        if not vals:
            return None, None, len(vals)
        m = statistics.mean(vals)
        s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return m, s, len(vals)

    print("\n--- Sentence-level z-scored features: mean (std) by sentence_label ---")
    for field in SENTENCE_LEVEL_FEATURES:
        if field in skip_fields:
            continue
        zf = f"{field}_z"
        print(f"\n{zf}:")
        for label in ("human", "ai", "ai_edited"):
            label_rows = [r for r in zscored_rows if r["sentence_label"] == label]
            m, s, n = zmean_std(label_rows, zf)
            print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={n}")

    print("\n--- Essay-level z-scored features: mean (std) by essay_label (deduplicated) ---")
    for field in ESSAY_LEVEL_FEATURES:
        if field in skip_fields:
            continue
        zf = f"{field}_z"
        print(f"\n{zf}:")
        for label in ("human", "ai", "hybrid"):
            label_rows = [r for r in zscored_rows if r["essay_label"] == label]
            m, s, n = zmean_std(label_rows, zf, dedupe_essay=True)
            print(f"  {label:10s} mean={m:.4f}  std={s:.4f}  n={n}")

    # Re-confirm 7 directional checks in z-score form
    print("\n" + "=" * 70)
    print("RE-CONFIRMED DIRECTIONAL CHECKS (z-score form vs Step 4 raw form)")
    print("=" * 70)

    def sent_mean_z(field_z):
        h = [r[field_z] for r in zscored_rows if r["sentence_label"] == "human" and r.get(field_z) is not None]
        a = [r[field_z] for r in zscored_rows if r["sentence_label"] in ("ai", "ai_edited") and r.get(field_z) is not None]
        return statistics.mean(h), statistics.mean(a)

    def essay_mean_z(field_z, label):
        seen = {}
        for r in zscored_rows:
            if r["essay_label"] == label:
                seen.setdefault(r["essay_id"], r)
        vals = [r[field_z] for r in seen.values() if r.get(field_z) is not None]
        return statistics.mean(vals)

    checks_z = []
    if "perplexity" not in skip_fields:
        h, a = sent_mean_z("perplexity_z")
        checks_z.append(("1. perplexity_z", h, a, a < h, "ai LOWER"))
    if "logprob_variance" not in skip_fields:
        h, a = sent_mean_z("logprob_variance_z")
        checks_z.append(("2. logprob_variance_z", h, a, a < h, "ai LOWER"))
    if "sentence_length_variance" not in skip_fields:
        h, a = essay_mean_z("sentence_length_variance_z", "human"), essay_mean_z("sentence_length_variance_z", "ai")
        checks_z.append(("3. sentence_length_variance_z", h, a, a < h, "ai LOWER"))
    if "syntactic_depth_variance" not in skip_fields:
        h, a = essay_mean_z("syntactic_depth_variance_z", "human"), essay_mean_z("syntactic_depth_variance_z", "ai")
        checks_z.append(("4. syntactic_depth_variance_z", h, a, a < h, "ai LOWER"))
    if "perplexity_variance" not in skip_fields:
        h, a = essay_mean_z("perplexity_variance_z", "human"), essay_mean_z("perplexity_variance_z", "ai")
        checks_z.append(("5. perplexity_variance_z", h, a, a < h, "ai LOWER"))
    if "ngram_repeat_rate" not in skip_fields:
        h, a = essay_mean_z("ngram_repeat_rate_z", "human"), essay_mean_z("ngram_repeat_rate_z", "ai")
        checks_z.append(("6. ngram_repeat_rate_z", h, a, a > h, "ai HIGHER"))
    if "transition_repetition_rate" not in skip_fields:
        h, a = essay_mean_z("transition_repetition_rate_z", "human"), essay_mean_z("transition_repetition_rate_z", "ai")
        checks_z.append(("7. transition_repetition_rate_z", h, a, a > h, "ai HIGHER"))

    step4_matches = {
        "1. perplexity_z": True, "2. logprob_variance_z": True,
        "3. sentence_length_variance_z": True, "4. syntactic_depth_variance_z": True,
        "5. perplexity_variance_z": False, "6. ngram_repeat_rate_z": False,
        "7. transition_repetition_rate_z": False,
    }
    n_match = 0
    flipped = []
    for name, h_val, a_val, matches, expect in checks_z:
        n_match += matches
        step4_match = step4_matches[name]
        flip_flag = " <<< FLIPPED FROM STEP 4" if matches != step4_match else ""
        if matches != step4_match:
            flipped.append(name)
        print(f"\n{name}: human={h_val:.4f}, ai/ai_edited={a_val:.4f}")
        print(f"  Expected: {expect}. {'MATCH' if matches else 'NO MATCH'} "
              f"(Step 4 raw-value result: {'MATCH' if step4_match else 'NO MATCH'}){flip_flag}")

    print(f"\n{n_match}/7 checks matched in z-score form.")
    if flipped:
        print(f"\n!! FLAGGED: {len(flipped)} check(s) flipped direction vs Step 4: {flipped}")
    else:
        print("\nNo direction flips vs Step 4 - normalization is a pure linear transform, as expected.")

    # Distribution shape for flagged features 5, 6, 7
    print("\n" + "=" * 70)
    print("DISTRIBUTION SHAPE: flagged features 5, 6, 7 (essay-level, deduplicated)")
    print("=" * 70)
    for field in ["perplexity_variance", "ngram_repeat_rate", "transition_repetition_rate"]:
        if field in skip_fields:
            continue
        zf = f"{field}_z"
        seen = {}
        for r in zscored_rows:
            seen.setdefault(r["essay_id"], r)
        vals = [r[zf] for r in seen.values() if r.get(zf) is not None]
        vals_sorted = sorted(vals)
        print(f"\n{zf}: min={vals_sorted[0]:.4f}, max={vals_sorted[-1]:.4f}, "
              f"n={len(vals)}")
        extreme = [v for v in vals if abs(v) > 3]
        print(f"  |z| > 3 (potential outliers): {len(extreme)}/{len(vals)}")
        if extreme:
            top_extreme = sorted(
                [(r["essay_id"], r["essay_label"], r[zf]) for r in seen.values() if r.get(zf) is not None and abs(r[zf]) > 3],
                key=lambda x: -abs(x[2])
            )
            for eid, label, z in top_extreme[:5]:
                print(f"    {eid} ({label}): z={z:.4f}")


if __name__ == "__main__":
    main()
