"""
Phase 4, Steps 2 & 3: evaluate both saved models on the true, held-out test set for the
first time. No retraining, no retuning - whatever comes out is reported as-is.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
TEST_FEATURES_PATH = ROOT / "dataset" / "features_test_zscored.jsonl"
ESSAY_MODEL_PATH = ROOT / "dataset" / "logistic_regression_model.joblib"
SENTENCE_MODEL_PATH = ROOT / "dataset" / "sentence_level_model.joblib"

ESSAY_FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]
SENTENCE_FEATURES = ["perplexity_z", "logprob_variance_z"]

STEP7_VAL = {"human": 0.8535, "ai": 0.9987, "ai_edited": 0.2000}
STEP7_6_VAL = {"human": 0.4262, "ai": 0.6660, "ai_edited": 0.7128}


def load_rows():
    rows = [json.loads(l) for l in TEST_FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["target"] = 0 if r["sentence_label"] == "human" else 1
    return rows


def report_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[{label}] Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(f"  Confusion matrix [[TN,FP],[FN,TP]]: {cm.tolist()}")
    return acc, prec, rec, f1


def per_label_accuracy(rows, y_true, y_pred, val_comparison):
    labels = np.array([r["sentence_label"] for r in rows])
    print("  Per-label accuracy (vs. Step 7/7.6 validation):")
    results = {}
    for label in ("human", "ai", "ai_edited"):
        mask = labels == label
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results[label] = float(acc)
        old = val_comparison[label]
        print(f"    {label:10s} n={mask.sum():5d}  validation={old:.4f}  TEST={acc:.4f}  delta={acc - old:+.4f}")
    return results


def main():
    rows = load_rows()

    # =================================================================
    print("=" * 70)
    print("STEP 2: ESSAY-LEVEL MODEL ON TEST SET")
    print("=" * 70)

    essay_rows = [r for r in rows if all(r.get(f) is not None for f in ESSAY_FEATURES)]
    print(f"Usable rows (all 7 features present): {len(essay_rows)}/{len(rows)}")

    model = joblib.load(ESSAY_MODEL_PATH)
    X = np.array([[r[f] for f in ESSAY_FEATURES] for r in essay_rows])
    y_true = np.array([r["target"] for r in essay_rows])
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    print("\n--- Sentence-level (directly comparable to Step 7 validation) ---")
    report_metrics(y_true, y_pred, "essay-level model, sentence-level eval")
    essay_model_sentence_perlabel = per_label_accuracy(essay_rows, y_true, y_pred, STEP7_VAL)

    print("\n--- Essay-level aggregation ---")
    print("Method: mean predicted probability across all sentences in the essay, "
          "thresholded at 0.5. Step 7 never aggregated to essay level (it only trained/")
    print("evaluated at sentence granularity) - this is a new decision for this step. "
          "Mean probability chosen over majority vote because 5 of the 7 features are ")
    print("essay-wide constants (same value on every sentence), so per-sentence "
          "probabilities within one essay are already highly correlated; averaging is a ")
    print("natural, low-variance summary rather than an arbitrary vote.")

    by_essay_proba = {}
    by_essay_true = {}
    for r, p in zip(essay_rows, y_proba):
        by_essay_proba.setdefault(r["essay_id"], []).append(p)
        by_essay_true[r["essay_id"]] = 0 if r["essay_label"] == "human" else 1

    essay_ids = sorted(by_essay_proba.keys())
    essay_mean_proba = np.array([statistics.mean(by_essay_proba[e]) for e in essay_ids])
    essay_pred = (essay_mean_proba >= 0.5).astype(int)
    essay_true = np.array([by_essay_true[e] for e in essay_ids])

    report_metrics(essay_true, essay_pred, "essay-level model, ESSAY-level eval")

    essay_labels = []
    id_to_label = {r["essay_id"]: r["essay_label"] for r in essay_rows}
    for eid in essay_ids:
        essay_labels.append(id_to_label[eid])
    essay_labels = np.array(essay_labels)
    print("  Per-essay-label accuracy (essay-level aggregation):")
    for label in ("human", "ai", "hybrid"):
        mask = essay_labels == label
        if mask.sum() == 0:
            continue
        acc = accuracy_score(essay_true[mask], essay_pred[mask])
        print(f"    {label:10s} n={mask.sum():3d}  accuracy={acc:.4f}")

    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SENTENCE-LEVEL MODEL ON TEST SET")
    print("=" * 70)

    sent_rows = [r for r in rows if all(r.get(f) is not None for f in SENTENCE_FEATURES)]
    print(f"Usable rows (2 features present): {len(sent_rows)}/{len(rows)}")

    model2 = joblib.load(SENTENCE_MODEL_PATH)
    X2 = np.array([[r[f] for f in SENTENCE_FEATURES] for r in sent_rows])
    y_true2 = np.array([r["target"] for r in sent_rows])
    y_pred2 = model2.predict(X2)

    report_metrics(y_true2, y_pred2, "sentence-level model")
    sentence_model_perlabel = per_label_accuracy(sent_rows, y_true2, y_pred2, STEP7_6_VAL)

    print("\n--- THE key question: did ai_edited recall hold up on unseen essays? ---")
    old_ae = STEP7_6_VAL["ai_edited"]
    new_ae = sentence_model_perlabel.get("ai_edited")
    if new_ae is not None:
        delta = new_ae - old_ae
        if abs(delta) < 0.05:
            verdict = "HELD UP - within 5 points of the validation number, not an artifact of that specific slice."
        elif delta < -0.15:
            verdict = "DID NOT HOLD UP - dropped substantially on genuinely unseen data, validation number was likely optimistic."
        elif delta < 0:
            verdict = "PARTIALLY HELD UP - dropped noticeably but still represents real improvement over the essay-level model's 20%."
        else:
            verdict = "HELD UP OR IMPROVED on unseen data."
        print(f"Validation: {old_ae:.2%}  Test: {new_ae:.2%}  Delta: {delta:+.2%}")
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
