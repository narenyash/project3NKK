"""
Phase 4, Step 5: ESL fairness check. Loads a sample of dataset/ellipse_fairness.jsonl
(all labeled human, non-native English writers), runs the same feature-extraction and
normalization pipeline, and reports the false-positive rate for both models compared
against the main test set's human false-positive rate.

Sampling disclosure: the full ELLIPSE corpus is 3,911 essays - many times larger than our
own 60-essay test set and prohibitively slow to fully process here. A seeded random
sample of 300 essays (~6,000+ sentences, a substantial sample, comparable in scale to a
meaningful fraction of our own train set) is used instead. This is disclosed explicitly,
not silently substituted for the full corpus.
"""
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np

import features as F
from text_utils import split_sentences, word_count

ROOT = Path(__file__).resolve().parent.parent
ELLIPSE_PATH = ROOT / "dataset" / "ellipse_fairness.jsonl"
BASELINE_PATH = ROOT / "dataset" / "corpus_baseline.json"
OUT_PATH = ROOT / "dataset" / "features_ellipse_sample.jsonl"
ESSAY_MODEL_PATH = ROOT / "dataset" / "logistic_regression_model.joblib"
SENTENCE_MODEL_PATH = ROOT / "dataset" / "sentence_level_model.joblib"

SAMPLE_SIZE = 300
SEED = 42
WORD_CAP = 1000  # same convention as length_normalize.py, applied for consistency

ESSAY_FEATURES = [
    "perplexity_z", "logprob_variance_z", "sentence_length_variance_z",
    "syntactic_depth_variance_z", "perplexity_variance_z", "ngram_repeat_rate_z",
    "transition_repetition_rate_z",
]
SENTENCE_FEATURES = ["perplexity_z", "logprob_variance_z"]
RAW_FEATURES = ["perplexity", "logprob_variance", "sentence_length_variance",
                "syntactic_depth_variance", "perplexity_variance", "ngram_repeat_rate",
                "transition_repetition_rate"]

# From Step 2/3 (main test set, genuine human essays)
MAIN_TEST_HUMAN_FPR = {
    "essay_model_sentence_level": 1 - 0.8352,
    "essay_model_essay_level": 1 - 0.8000,
    "sentence_model": 1 - 0.4365,
}


def truncate_to_cap(text):
    if word_count(text) <= WORD_CAP:
        return text
    sentences = split_sentences(text)
    kept, total = [], 0
    for s in sentences:
        w = word_count(s)
        if total + w > WORD_CAP and kept:
            break
        kept.append(s)
        total += w
    return " ".join(kept)


def main():
    all_rows = [json.loads(l) for l in ELLIPSE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("=" * 70)
    print("PHASE 4 STEP 5: ESL FAIRNESS CHECK")
    print("=" * 70)
    print(f"Full ELLIPSE corpus: {len(all_rows)} essays. Sampling {SAMPLE_SIZE} (seed={SEED}) "
          f"for tractability - disclosed, not a silent substitution.")

    rng = random.Random(SEED)
    sample = rng.sample(all_rows, SAMPLE_SIZE)

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    print("Loading GPT-2...")
    F.get_model()

    feature_rows = []
    for i, essay in enumerate(sample, 1):
        text = truncate_to_cap(essay["text"].strip())
        sentences_text = split_sentences(text)
        if not sentences_text:
            continue

        perplexities, logprob_vars = [], []
        for idx, s in enumerate(sentences_text):
            start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
            context = " ".join(sentences_text[start:idx])
            token_lps = F.sentence_token_logprobs(s, context)
            perplexities.append(F.sentence_perplexity(token_lps))
            logprob_vars.append(F.token_logprob_variance(token_lps))

        burst = F.essay_burstiness(sentences_text, perplexities)
        ngram = F.essay_ngram_repetition(sentences_text)

        for idx, s in enumerate(sentences_text):
            raw = {
                "essay_id": essay["id"],
                "sentence_idx": idx,
                "sentence_text": s,
                "perplexity": perplexities[idx],
                "logprob_variance": logprob_vars[idx],
                "sentence_length_variance": burst["sentence_length_variance"],
                "syntactic_depth_variance": burst["syntactic_depth_variance"],
                "perplexity_variance": burst["perplexity_variance"],
                "ngram_repeat_rate": ngram["ngram_repeat_rate"],
                "transition_repetition_rate": ngram["transition_repetition_rate"],
            }
            for field in RAW_FEATURES:
                v = raw[field]
                m, sd = baseline[field]["mean"], baseline[field]["std"]
                if v is None or sd == 0:
                    raw[f"{field}_z"] = None
                    continue
                z = max(-5.0, min(5.0, (v - m) / sd))
                raw[f"{field}_z"] = z
            feature_rows.append(raw)

        if i % 50 == 0 or i == len(sample):
            print(f"[{i}/{len(sample)}] essays done", flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in feature_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(feature_rows)} sentence rows ({len(sample)} essays) to {OUT_PATH.relative_to(ROOT)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING BOTH MODELS ON ELLIPSE SAMPLE")
    print("=" * 70)

    essay_model = joblib.load(ESSAY_MODEL_PATH)
    sentence_model = joblib.load(SENTENCE_MODEL_PATH)

    usable1 = [r for r in feature_rows if all(r.get(f"{f}_z") is not None for f in RAW_FEATURES)]
    X1 = np.array([[r[f"{f}_z"] for f in RAW_FEATURES] for r in usable1])
    proba1 = essay_model.predict_proba(X1)[:, 1]
    pred1 = (proba1 >= 0.5).astype(int)
    sentence_fpr_essay_model = pred1.mean()
    print(f"\nEssay-level model, SENTENCE-level: {pred1.sum()}/{len(pred1)} ELL sentences "
          f"flagged as AI. FPR = {sentence_fpr_essay_model:.4f}")

    by_essay_proba = {}
    for r, p in zip(usable1, proba1):
        by_essay_proba.setdefault(r["essay_id"], []).append(p)
    essay_mean_proba = np.array([np.mean(v) for v in by_essay_proba.values()])
    essay_pred = (essay_mean_proba >= 0.5).astype(int)
    essay_fpr_essay_model = essay_pred.mean()
    print(f"Essay-level model, ESSAY-level: {essay_pred.sum()}/{len(essay_pred)} ELL essays "
          f"flagged as AI. FPR = {essay_fpr_essay_model:.4f}")

    usable2 = [r for r in feature_rows if all(r.get(f"{f}_z") is not None for f in ["perplexity", "logprob_variance"])]
    X2 = np.array([[r["perplexity_z"], r["logprob_variance_z"]] for r in usable2])
    pred2 = sentence_model.predict(X2)
    sentence_fpr_sentence_model = pred2.mean()
    print(f"Sentence-level model: {pred2.sum()}/{len(pred2)} ELL sentences flagged as AI. "
          f"FPR = {sentence_fpr_sentence_model:.4f}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON: ELL FALSE-POSITIVE RATE vs. MAIN TEST SET HUMAN FALSE-POSITIVE RATE")
    print("=" * 70)
    comparisons = [
        ("Essay model, sentence-level", sentence_fpr_essay_model, MAIN_TEST_HUMAN_FPR["essay_model_sentence_level"]),
        ("Essay model, essay-level", essay_fpr_essay_model, MAIN_TEST_HUMAN_FPR["essay_model_essay_level"]),
        ("Sentence-level model", sentence_fpr_sentence_model, MAIN_TEST_HUMAN_FPR["sentence_model"]),
    ]
    for name, ell_fpr, main_fpr in comparisons:
        delta = ell_fpr - main_fpr
        flag = "ELEVATED on ESL writing" if delta > 0.05 else ("LOWER on ESL writing" if delta < -0.05 else "COMPARABLE")
        print(f"\n{name}:")
        print(f"  Main test set (genuine human) FPR: {main_fpr:.4f}")
        print(f"  ELLIPSE (ESL) FPR:                 {ell_fpr:.4f}")
        print(f"  Delta: {delta:+.4f} -> {flag}")


if __name__ == "__main__":
    main()
