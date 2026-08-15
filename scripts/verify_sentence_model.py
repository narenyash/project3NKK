"""Sanity-check the Step 7.6 sentence-level model: per-essay ai_edited accuracy breakdown."""
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPPED_PATH = ROOT / "dataset" / "features_train_clipped.jsonl"
MODEL_PATH = ROOT / "dataset" / "sentence_level_model.joblib"

FEATURES = ["perplexity_z", "logprob_variance_z"]
SEED = 42
VAL_FRACTION = 0.2


def main():
    rows = [json.loads(l) for l in CLIPPED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["target"] = 0 if r["sentence_label"] == "human" else 1

    essays = {}
    for r in rows:
        essays.setdefault(r["essay_id"], r["essay_label"])
    by_essay_label = {"human": [], "ai": [], "hybrid": []}
    for eid, elabel in essays.items():
        by_essay_label[elabel].append(eid)
    rng = random.Random(SEED)
    val_essay_ids = set()
    for elabel, eids in by_essay_label.items():
        eids = sorted(eids)
        rng.shuffle(eids)
        n_val = round(len(eids) * VAL_FRACTION)
        val_essay_ids.update(eids[:n_val])

    model = joblib.load(MODEL_PATH)

    val_ae_rows = [r for r in rows if r["essay_id"] in val_essay_ids and r["sentence_label"] == "ai_edited"
                   and all(r.get(f) is not None for f in FEATURES)]

    by_essay = {}
    for r in val_ae_rows:
        by_essay.setdefault(r["essay_id"], []).append(r)

    print(f"{'essay_id':10s} {'n_ai_edited':>12s} {'n_correct':>10s} {'accuracy':>10s}")
    total_correct, total_n = 0, 0
    n_essays_above_50 = 0
    for eid, ae_rows in sorted(by_essay.items()):
        X = np.array([[r[f] for f in FEATURES] for r in ae_rows])
        y_true = np.array([1] * len(ae_rows))  # ai_edited is always target=1
        y_pred = model.predict(X)
        correct = (y_pred == y_true).sum()
        acc = correct / len(ae_rows)
        total_correct += correct
        total_n += len(ae_rows)
        if acc >= 0.5:
            n_essays_above_50 += 1
        print(f"{eid:10s} {len(ae_rows):12d} {correct:10d} {acc:10.2%}")

    print(f"\nOverall: {total_correct}/{total_n} = {total_correct/total_n:.2%}")
    print(f"Essays with >=50% ai_edited sentences correctly flagged: {n_essays_above_50}/{len(by_essay)}")

    # Also check: what's the raw perplexity_z distribution for ai_edited vs human within
    # these SAME validation hybrid essays, to see if the separation is a clean, sane gap
    # or a weird artifact.
    import statistics
    ae_ppl_z = [r["perplexity_z"] for r in val_ae_rows]
    human_in_hybrid = [r for r in rows if r["essay_id"] in val_essay_ids and r["essay_label"] == "hybrid"
                        and r["sentence_label"] == "human" and r.get("perplexity_z") is not None]
    h_ppl_z = [r["perplexity_z"] for r in human_in_hybrid]
    print(f"\nperplexity_z within validation hybrid essays:")
    print(f"  ai_edited: mean={statistics.mean(ae_ppl_z):.4f}, median={statistics.median(ae_ppl_z):.4f}, n={len(ae_ppl_z)}")
    print(f"  human:     mean={statistics.mean(h_ppl_z):.4f}, median={statistics.median(h_ppl_z):.4f}, n={len(h_ppl_z)}")


if __name__ == "__main__":
    main()
