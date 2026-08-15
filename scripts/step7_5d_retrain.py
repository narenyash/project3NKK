"""
Phase 2, Step 7.5d: retrain LogisticRegression with the 4 approved features + 3 new
local features (10 total), using the EXACT SAME essay-stratified train-sub/validation
split as Step 7 (same seed=42, same algorithm - reproducible since essay/label
assignments haven't changed).
"""
import json
import random
import warnings
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
CLIPPED_PATH = ROOT / "dataset" / "features_train_clipped.jsonl"
LOCAL_CLIPPED_PATH = ROOT / "dataset" / "features_train_with_local_clipped.jsonl"

# 4 approved Step 2 features + 3 new local features
FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z", "syntactic_depth_variance_z",
    "local_perplexity_deviation_z", "local_length_deviation_z", "local_depth_deviation_z",
]
SEED = 42
VAL_FRACTION = 0.2

# Step 7 (post Tab-N-fix) results, for direct comparison
STEP7 = {
    "accuracy": 0.8398, "precision": 0.9168, "recall": 0.8328, "f1": 0.8728,
    "per_label": {"human": 0.8535, "ai": 0.9987, "ai_edited": 0.2000},
}


def main():
    base_rows = {(r["essay_id"], r["sentence_idx"]): r for r in
                 (json.loads(l) for l in CLIPPED_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}
    local_rows = [json.loads(l) for l in LOCAL_CLIPPED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    merged = []
    for r in local_rows:
        key = (r["essay_id"], r["sentence_idx"])
        base = base_rows.get(key)
        if base is None:
            continue
        m = dict(base)
        m["local_perplexity_deviation_z"] = r.get("local_perplexity_deviation_z")
        m["local_length_deviation_z"] = r.get("local_length_deviation_z")
        m["local_depth_deviation_z"] = r.get("local_depth_deviation_z")
        m["target"] = 0 if m["sentence_label"] == "human" else 1
        merged.append(m)

    print("=" * 70)
    print("STEP 7.5d: RETRAIN WITH 4 APPROVED + 3 NEW LOCAL FEATURES (7 total)")
    print("=" * 70)
    print(f"Merged {len(merged)} rows.")

    essays = {}
    for r in merged:
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

    def has_all_features(r):
        return all(r.get(f) is not None for f in FEATURES)

    train_sub_rows = [r for r in merged if r["essay_id"] not in val_essay_ids and has_all_features(r)]
    val_rows = [r for r in merged if r["essay_id"] in val_essay_ids and has_all_features(r)]
    print(f"Validation essays: {len(val_essay_ids)} (same 48 as Step 7)")
    print(f"Train-sub rows: {len(train_sub_rows)}, validation rows: {len(val_rows)}")

    X_train = np.array([[r[f] for f in FEATURES] for r in train_sub_rows])
    y_train = np.array([r["target"] for r in train_sub_rows])
    X_val = np.array([[r[f] for f in FEATURES] for r in val_rows])
    y_val = np.array([r["target"] for r in val_rows])

    max_iter = 1000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=SEED)
        model.fit(X_train, y_train)
        conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    print(f"\nConverged: {'YES, max_iter=' + str(max_iter) if not conv else 'NO - ' + str(conv)}")

    print("\n--- Coefficients (sorted by magnitude) ---")
    coefs = sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1]))
    print(f"Intercept: {model.intercept_[0]:.4f}\n")
    for f, c in coefs:
        tag = " <- NEW (Step 7.5)" if "local" in f else ""
        print(f"  {f:32s} {c:+10.4f}{tag}")

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n--- Validation metrics ---")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm.tolist()}")

    print(f"\nStep 7 (7 features, post Tab-N-fix): "
          f"acc={STEP7['accuracy']:.4f}, prec={STEP7['precision']:.4f}, "
          f"rec={STEP7['recall']:.4f}, f1={STEP7['f1']:.4f}")
    print(f"Step 7.5 (7 features incl. 3 local):  "
          f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    print(f"Delta accuracy: {acc - STEP7['accuracy']:+.4f}")

    print("\n--- Per-label accuracy: Step 7 vs Step 7.5 ---")
    val_labels = np.array([r["sentence_label"] for r in val_rows])
    for label in ("human", "ai", "ai_edited"):
        mask = val_labels == label
        if mask.sum() == 0:
            continue
        label_acc = accuracy_score(y_val[mask], y_pred[mask])
        old = STEP7["per_label"][label]
        print(f"  {label:10s} n={mask.sum():5d}  Step7={old:.4f}  Step7.5={label_acc:.4f}  "
              f"delta={label_acc - old:+.4f}")


if __name__ == "__main__":
    main()
