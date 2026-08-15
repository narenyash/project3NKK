"""
Phase 2, Step 7.5c: extend corpus_baseline.json with the 3 new local features and apply
the same [-5,5] clipping convention. Unlike the essay-level features from Step 5, these 3
are genuinely SENTENCE-level (vary within an essay), so the baseline uses all
sentence_label == "human" rows directly - same treatment as perplexity_z/logprob_variance_z
in Step 5, not the essay_label deduplication used for the essay-wide features.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
WITH_LOCAL_PATH = ROOT / "dataset" / "features_train_with_local.jsonl"
BASELINE_PATH = ROOT / "dataset" / "corpus_baseline.json"
OUT_PATH = ROOT / "dataset" / "features_train_with_local_clipped.jsonl"

NEW_FEATURES = ["local_perplexity_deviation", "local_length_deviation", "local_depth_deviation"]
CLIP_LO, CLIP_HI = -5.0, 5.0


def main():
    rows = [json.loads(l) for l in WITH_LOCAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    human_rows = [r for r in rows if r["sentence_label"] == "human"]

    print("=" * 70)
    print("STEP 7.5c: EXTEND BASELINE, NORMALIZE, CLIP")
    print("=" * 70)

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for field in NEW_FEATURES:
        vals = [r[field] for r in human_rows if r.get(field) is not None]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals)
        baseline[field] = {"mean": mean, "std": std}
        print(f"{field} (sentence-level, n={len(vals)}): mean={mean:.4f}, std={std:.4f}")

    with BASELINE_PATH.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nExtended {BASELINE_PATH.relative_to(ROOT)} (10 features total now)")

    clip_counts = {f: 0 for f in NEW_FEATURES}
    for r in rows:
        for field in NEW_FEATURES:
            v = r.get(field)
            if v is None:
                continue
            z = (v - baseline[field]["mean"]) / baseline[field]["std"] if baseline[field]["std"] > 0 else None
            if z is None:
                r[f"{field}_z"] = None
                continue
            if z < CLIP_LO:
                z = CLIP_LO
                clip_counts[field] += 1
            elif z > CLIP_HI:
                z = CLIP_HI
                clip_counts[field] += 1
            r[f"{field}_z"] = z

    print(f"\nClipping counts: {clip_counts}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
