"""
Phase 2, Step 7: train the real, interpretable logistic regression detector.
Uses all 7 z-scored (clipped) features from dataset/features_train_clipped.jsonl. This
is the step where features 5-7's fate (perplexity_variance, ngram_repeat_rate,
transition_repetition_rate) is finally decided by their TRAINED coefficients, not by raw
means or guesswork. Trains/evaluates only within the train split - the held-out test set
is untouched here and reserved for a later, separate evaluation phase.
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)

ROOT = Path(__file__).resolve().parent.parent
CLIPPED_PATH = ROOT / "dataset" / "features_train_clipped.jsonl"
MODEL_JOBLIB_PATH = ROOT / "dataset" / "logistic_regression_model.joblib"
MODEL_JSON_PATH = ROOT / "dataset" / "logistic_regression_model.json"

FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]
ESTABLISHED_DIRECTION = {
    "perplexity_z": "matches",
    "logprob_variance_z": "matches",
    "sentence_length_variance_z": "matches",
    "syntactic_depth_variance_z": "matches",
    "perplexity_variance_z": "inverted",
    "ngram_repeat_rate_z": "non-separating",
    "transition_repetition_rate_z": "non-separating",
}

SEED = 42
VAL_FRACTION = 0.2

# Step 6 baseline, for direct comparison
BASELINE = {"accuracy": 0.7656, "precision": 0.8544, "recall": 0.7862}


def main():
    rows = [json.loads(l) for l in CLIPPED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 7a: TARGET LABEL FRAMING")
    print("=" * 70)
    print("Binary framing confirmed: human=0, ai/ai_edited=1 ('AI-influenced' vs human).")
    print("This matches the practical app need (flag AI involvement at all) and keeps the")
    print("model consistent with every directional check run since Step 3.\n")

    for r in rows:
        r["target"] = 0 if r["sentence_label"] == "human" else 1

    from collections import Counter
    label_counts = Counter(r["sentence_label"] for r in rows)
    target_counts = Counter(r["target"] for r in rows)
    print(f"Class counts (sentence_label): {dict(label_counts)}")
    print(f"Class counts (binary target): human(0)={target_counts[0]}, "
          f"ai/ai_edited(1)={target_counts[1]}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7b: TRAIN/VALIDATION SPLIT (within train split, essay-stratified)")
    print("=" * 70)

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

    train_sub_rows = [r for r in rows if r["essay_id"] not in val_essay_ids and all(r.get(f) is not None for f in FEATURES)]
    val_rows = [r for r in rows if r["essay_id"] in val_essay_ids and all(r.get(f) is not None for f in FEATURES)]
    dropped_for_missing = [r for r in rows if any(r.get(f) is None for f in FEATURES)]

    print(f"Essays: {len(essays)} total ({len(by_essay_label['human'])} human, "
          f"{len(by_essay_label['ai'])} ai, {len(by_essay_label['hybrid'])} hybrid)")
    print(f"Validation essays: {len(val_essay_ids)} "
          f"(no essay appears in both train_sub and val - split at essay level)")
    print(f"Train-sub sentence rows: {len(train_sub_rows)}")
    print(f"Validation sentence rows: {len(val_rows)}")
    if dropped_for_missing:
        print(f"Dropped {len(dropped_for_missing)} rows with a missing feature value "
              f"(same as Step 6 - not imputed).")

    train_sub_label_counts = Counter(r["sentence_label"] for r in train_sub_rows)
    val_label_counts = Counter(r["sentence_label"] for r in val_rows)
    print(f"Train-sub by sentence_label: {dict(train_sub_label_counts)}")
    print(f"Validation by sentence_label: {dict(val_label_counts)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7c: TRAIN LOGISTIC REGRESSION")
    print("=" * 70)

    X_train = np.array([[r[f] for f in FEATURES] for r in train_sub_rows])
    y_train = np.array([r["target"] for r in train_sub_rows])
    X_val = np.array([[r[f] for f in FEATURES] for r in val_rows])
    y_val = np.array([r["target"] for r in val_rows])

    print(f"class_weight='balanced' applied (label imbalance: "
          f"{dict(Counter(y_train.tolist()))} in train-sub, human(0) vs ai/ai_edited(1))")

    max_iter = 1000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=SEED)
        model.fit(X_train, y_train)
        conv_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]

    if conv_warnings:
        print(f"!! Solver did NOT converge within max_iter={max_iter}. Warning(s):")
        for w in conv_warnings:
            print(f"   {w.message}")
        print("   Retrying with max_iter=5000...")
        max_iter = 5000
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=SEED)
            model.fit(X_train, y_train)
            conv_warnings2 = [w for w in caught2 if issubclass(w.category, ConvergenceWarning)]
        if conv_warnings2:
            print(f"!! STILL did not converge at max_iter={max_iter}. Reporting as-is, not hiding this.")
        else:
            print(f"   Converged at max_iter={max_iter}.")
    else:
        print(f"Converged cleanly at max_iter={max_iter}. No solver warnings.")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7d: COEFFICIENT REPORT")
    print("=" * 70)

    coefs = list(zip(FEATURES, model.coef_[0]))
    coefs_sorted = sorted(coefs, key=lambda x: -abs(x[1]))
    print(f"Intercept: {model.intercept_[0]:.4f}\n")
    print(f"{'feature':32s} {'coef':>10s}  established direction")
    for f, c in coefs_sorted:
        print(f"{f:32s} {c:+10.4f}  {ESTABLISHED_DIRECTION[f]}")

    all_mags = [abs(c) for _, c in coefs]
    max_mag = max(all_mags)
    print(f"\nFor reference: max |coefficient| across all 7 features = {max_mag:.4f}")

    print("\n--- Explicit calls for features 5, 6, 7 ---")
    calls = {}
    for f in ["perplexity_variance_z", "ngram_repeat_rate_z", "transition_repetition_rate_z"]:
        c = dict(coefs)[f]
        rel_mag = abs(c) / max_mag
        if rel_mag < 0.15:
            call = "DROP (near-zero relative to strongest feature)"
        else:
            sign_note = ("useful INVERTED signal - opposite sign to the original "
                          "hypothesis, but not the same as no signal" if ESTABLISHED_DIRECTION[f] == "inverted"
                          else "stable, non-trivial coefficient despite not separating on raw means")
            call = f"KEEP - {sign_note}"
        calls[f] = call
        print(f"  {f}: coef={c:+.4f} ({rel_mag:.1%} of strongest feature's magnitude) -> {call}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7e: VALIDATION EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]] (positive=ai/ai_edited):")
    print(f"  {cm.tolist()}")

    print(f"\nStep 6 baseline (4-feature weighted sum): "
          f"accuracy={BASELINE['accuracy']:.4f}, precision={BASELINE['precision']:.4f}, "
          f"recall={BASELINE['recall']:.4f}")
    print(f"Step 7 model (7-feature logistic regression): "
          f"accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}")
    delta_acc = acc - BASELINE["accuracy"]
    print(f"Accuracy delta: {delta_acc:+.4f} ({'BETTER' if delta_acc > 0.01 else 'WORSE' if delta_acc < -0.01 else 'ABOUT THE SAME'})")

    print("\n--- Per-label accuracy breakdown (the ai_edited weak-spot check) ---")
    val_sentence_labels = np.array([r["sentence_label"] for r in val_rows])
    for label in ("human", "ai", "ai_edited"):
        mask = val_sentence_labels == label
        if mask.sum() == 0:
            continue
        label_true = y_val[mask]
        label_pred = y_pred[mask]
        label_acc = accuracy_score(label_true, label_pred)
        print(f"  {label:10s} n={mask.sum():5d}  accuracy={label_acc:.4f}")

    ai_edited_mask = val_sentence_labels == "ai_edited"
    ai_mask = val_sentence_labels == "ai"
    if ai_edited_mask.sum() > 0 and ai_mask.sum() > 0:
        ai_edited_acc = accuracy_score(y_val[ai_edited_mask], y_pred[ai_edited_mask])
        ai_acc = accuracy_score(y_val[ai_mask], y_pred[ai_mask])
        gap = ai_acc - ai_edited_acc
        print(f"\nai_edited vs ai accuracy gap: {gap:+.4f} "
              f"({'GAP PERSISTS' if gap > 0.1 else 'GAP LARGELY CLOSED' if gap < 0.03 else 'GAP NARROWED BUT PRESENT'})")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7f: SAVE MODEL")
    print("=" * 70)

    joblib.dump(model, MODEL_JOBLIB_PATH)
    print(f"Wrote {MODEL_JOBLIB_PATH.relative_to(ROOT)} (reloadable via joblib.load(), no retraining needed)")

    model_json = {
        "features": FEATURES,
        "coefficients": {f: float(c) for f, c in coefs},
        "intercept": float(model.intercept_[0]),
        "class_weight": "balanced",
        "target_framing": "human=0, ai/ai_edited=1",
        "trained_on": "train split only (essay-stratified 80/20 sub-split, val held out within train)",
        "seed": SEED,
    }
    with MODEL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)
    print(f"Wrote {MODEL_JSON_PATH.relative_to(ROOT)} (human-readable coefficients, "
          f"usable for inference without sklearn: score = sigmoid(intercept + sum(coef_i * z_i)))")


if __name__ == "__main__":
    main()
