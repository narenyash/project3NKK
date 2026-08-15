"""
Phase 2, Step 7.6: two-tier model split. The Step 7 essay-level model (all 4 approved
features) stays untouched and is kept for essay-level scoring - it's genuinely strong
(99.7% ai, 85% human) and the diagnosis showed its dominant coefficients are essay-wide
constants that actively swamp sentence-level signal for the ai_edited case.

This trains a SEPARATE, sentence-level-ONLY model using just perplexity_z and
logprob_variance_z - the only 2 features confirmed to vary sentence-to-sentence with a
correctly-directioned, validated coefficient. Nothing essay-wide competes for weight here.

Same essay-stratified train-sub/validation split as Step 7 (48 held-out essays, seed=42).
ai and ai_edited accuracy are reported SEPARATELY throughout, not collapsed into one "AI"
number - per the methodological flag that ai-labeled sentences sit in AI-generated
context while ai_edited sentences sit in human context, which may be two subtly
different sub-tasks for a sentence-level model.
"""
import json
import random
import warnings
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
CLIPPED_PATH = ROOT / "dataset" / "features_train_clipped.jsonl"
MODEL_JOBLIB_PATH = ROOT / "dataset" / "sentence_level_model.joblib"
MODEL_JSON_PATH = ROOT / "dataset" / "sentence_level_model.json"

FEATURES = ["perplexity_z", "logprob_variance_z"]
SEED = 42
VAL_FRACTION = 0.2

# Step 7 essay-level model's per-label results, for direct comparison
STEP7_ESSAY_LEVEL = {
    "accuracy": 0.8398, "precision": 0.9168, "recall": 0.8328, "f1": 0.8728,
    "per_label": {"human": 0.8535, "ai": 0.9987, "ai_edited": 0.2000},
}


def main():
    rows = [json.loads(l) for l in CLIPPED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["target"] = 0 if r["sentence_label"] == "human" else 1

    print("=" * 70)
    print("STEP 7.6: SENTENCE-LEVEL-ONLY MODEL (perplexity_z + logprob_variance_z)")
    print("=" * 70)
    print("Essay-level model (Step 7, 4 features) is UNTOUCHED and remains the essay-")
    print("level scorer. This is a SEPARATE model for sentence-level flagging only.\n")

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

    def has_features(r):
        return all(r.get(f) is not None for f in FEATURES)

    train_sub_rows = [r for r in rows if r["essay_id"] not in val_essay_ids and has_features(r)]
    val_rows = [r for r in rows if r["essay_id"] in val_essay_ids and has_features(r)]
    print(f"Validation essays: {len(val_essay_ids)} (same 48 as Step 7)")
    print(f"Train-sub rows: {len(train_sub_rows)}, validation rows: {len(val_rows)}")

    X_train = np.array([[r[f] for f in FEATURES] for r in train_sub_rows])
    y_train = np.array([r["target"] for r in train_sub_rows])
    X_val = np.array([[r[f] for f in FEATURES] for r in val_rows])
    y_val = np.array([r["target"] for r in val_rows])

    from collections import Counter
    print(f"class_weight='balanced' applied (train-sub label counts: {dict(Counter(y_train.tolist()))})")

    max_iter = 1000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=SEED)
        model.fit(X_train, y_train)
        conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    print(f"Converged: {'YES' if not conv else 'NO - ' + str(conv)}")

    print("\n--- Coefficients ---")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    for f, c in zip(FEATURES, model.coef_[0]):
        print(f"  {f:20s} {c:+.4f}")

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n--- Aggregate validation metrics (human vs ai+ai_edited collapsed) ---")
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm.tolist()}")

    print("\n--- Per-label accuracy: ai and ai_edited reported SEPARATELY (not collapsed) ---")
    val_labels = np.array([r["sentence_label"] for r in val_rows])
    per_label_results = {}
    for label in ("human", "ai", "ai_edited"):
        mask = val_labels == label
        if mask.sum() == 0:
            continue
        label_acc = accuracy_score(y_val[mask], y_pred[mask])
        per_label_results[label] = float(label_acc)
        old = STEP7_ESSAY_LEVEL["per_label"][label]
        print(f"  {label:10s} n={mask.sum():5d}  essay-level(Step7)={old:.4f}  "
              f"sentence-level(Step7.6)={label_acc:.4f}  delta={label_acc - old:+.4f}")

    print("\n--- Verdict ---")
    ai_edited_new = per_label_results.get("ai_edited")
    if ai_edited_new is not None:
        if ai_edited_new >= 0.70:
            verdict = "STRONG improvement - unexpectedly good, worth double-checking for leakage"
        elif ai_edited_new >= 0.40:
            verdict = "PARTIAL improvement - real lift, still well short of the essay-level model's performance on ai/human"
        elif ai_edited_new > 0.20:
            verdict = "MODEST improvement - some lift but limited"
        else:
            verdict = "NO improvement - the decoupling alone did not help"
        print(f"ai_edited accuracy: 20.00% (Step 7) -> {ai_edited_new:.2%} (Step 7.6). {verdict}")

    # -----------------------------------------------------------------
    joblib.dump(model, MODEL_JOBLIB_PATH)
    model_json = {
        "features": FEATURES,
        "coefficients": {f: float(c) for f, c in zip(FEATURES, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "class_weight": "balanced",
        "target_framing": "human=0, ai/ai_edited=1 (sentence-level scorer, use alongside the essay-level model)",
        "seed": SEED,
        "purpose": "Sentence-level-only AI-touch flag, decoupled from essay-wide features per the Step 7 "
                   "diagnosis that essay-wide coefficients swamp sentence-level signal for ai_edited detection.",
    }
    with MODEL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)
    print(f"\nSaved {MODEL_JOBLIB_PATH.relative_to(ROOT)} and {MODEL_JSON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
