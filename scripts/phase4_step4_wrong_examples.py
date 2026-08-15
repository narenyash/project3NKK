"""
Phase 4, Step 4: find confidently-wrong examples across both models on the test set -
high predicted probability, wrong label. Report text, probability, features, and analysis.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_FEATURES_PATH = ROOT / "dataset" / "features_test_zscored.jsonl"
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"
ESSAY_MODEL_PATH = ROOT / "dataset" / "logistic_regression_model.joblib"
SENTENCE_MODEL_PATH = ROOT / "dataset" / "sentence_level_model.joblib"

ESSAY_FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]
SENTENCE_FEATURES = ["perplexity_z", "logprob_variance_z"]
RAW_FEATURES = ["perplexity", "logprob_variance", "sentence_length_variance",
                "syntactic_depth_variance", "perplexity_variance", "ngram_repeat_rate",
                "transition_repetition_rate"]


def main():
    rows = [json.loads(l) for l in TEST_FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    sent_text = {(r["essay_id"], r["sentence_idx"]): r["sentence_text"]
                 for r in (json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}
    for r in rows:
        r["target"] = 0 if r["sentence_label"] == "human" else 1
        r["text"] = sent_text.get((r["essay_id"], r["sentence_idx"]), "")

    essay_model = joblib.load(ESSAY_MODEL_PATH)
    sentence_model = joblib.load(SENTENCE_MODEL_PATH)

    essay_rows = [r for r in rows if all(r.get(f) is not None for f in ESSAY_FEATURES)]
    X1 = np.array([[r[f] for f in ESSAY_FEATURES] for r in essay_rows])
    proba1 = essay_model.predict_proba(X1)[:, 1]
    for r, p in zip(essay_rows, proba1):
        r["essay_model_proba"] = float(p)

    sent_rows = [r for r in rows if all(r.get(f) is not None for f in SENTENCE_FEATURES)]
    X2 = np.array([[r[f] for f in SENTENCE_FEATURES] for r in sent_rows])
    proba2 = sentence_model.predict_proba(X2)[:, 1]
    for r, p in zip(sent_rows, proba2):
        r["sentence_model_proba"] = float(p)

    print("=" * 70)
    print("MOST CONFIDENTLY WRONG - ESSAY-LEVEL MODEL")
    print("=" * 70)
    # human misclassified as AI with high confidence
    wrong_human = sorted([r for r in essay_rows if r["target"] == 0], key=lambda r: -r["essay_model_proba"])[:5]
    print("\nTop 5 human sentences the essay-level model was MOST confident were AI:")
    for r in wrong_human:
        print(f"\n  [{r['essay_id']}#{r['sentence_idx']}] proba(AI)={r['essay_model_proba']:.4f} true=human")
        print(f"    text: {r['text'][:150]!r}")
        print(f"    raw features: " + ", ".join(f"{f}={r[f]:.2f}" if r[f] is not None else f"{f}=None" for f in RAW_FEATURES))

    # ai/ai_edited misclassified as human with high confidence
    wrong_ai = sorted([r for r in essay_rows if r["target"] == 1], key=lambda r: r["essay_model_proba"])[:5]
    print("\nTop 5 AI/ai_edited sentences the essay-level model was MOST confident were human:")
    for r in wrong_ai:
        print(f"\n  [{r['essay_id']}#{r['sentence_idx']}] proba(AI)={r['essay_model_proba']:.4f} true={r['sentence_label']}")
        print(f"    text: {r['text'][:150]!r}")
        print(f"    raw features: " + ", ".join(f"{f}={r[f]:.2f}" if r[f] is not None else f"{f}=None" for f in RAW_FEATURES))

    print("\n" + "=" * 70)
    print("MOST CONFIDENTLY WRONG - SENTENCE-LEVEL MODEL")
    print("=" * 70)
    wrong_human2 = sorted([r for r in sent_rows if r["target"] == 0], key=lambda r: -r["sentence_model_proba"])[:5]
    print("\nTop 5 human sentences the sentence-level model was MOST confident were AI:")
    for r in wrong_human2:
        print(f"\n  [{r['essay_id']}#{r['sentence_idx']}] proba(AI)={r['sentence_model_proba']:.4f} true=human")
        print(f"    text: {r['text'][:150]!r}")
        print(f"    perplexity={r['perplexity']:.2f}, logprob_variance={r['logprob_variance']:.2f}")

    wrong_ai2 = sorted([r for r in sent_rows if r["target"] == 1], key=lambda r: r["sentence_model_proba"])[:5]
    print("\nTop 5 AI/ai_edited sentences the sentence-level model was MOST confident were human:")
    for r in wrong_ai2:
        print(f"\n  [{r['essay_id']}#{r['sentence_idx']}] proba(AI)={r['sentence_model_proba']:.4f} true={r['sentence_label']}")
        print(f"    text: {r['text'][:150]!r}")
        print(f"    perplexity={r['perplexity']:.2f}, logprob_variance={r['logprob_variance']:.2f}")


if __name__ == "__main__":
    main()
