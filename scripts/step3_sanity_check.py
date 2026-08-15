"""
Phase 2, Step 3: stratified sanity check (go/no-go gate) before full-scale extraction.

Runs all four Step 2 feature functions over a 30-essay stratified sample (10 human,
10 ai, 10 hybrid) drawn ONLY from the train split, and reports 7 directional checks.

Function 4 note: the spec asked to re-confirm a "widen the window" fix, but Function 4
was redesigned further in Step 2 (v4) into a whole-essay bigram/trigram repeat rate,
not a windowed diversity ratio - see features.py's Function 4 docstring/history comment
for the full v1-v4 story. That redesign is what's exercised here. Because the metric is
now a REPEAT rate (higher = more repetitive) rather than a DIVERSITY ratio (higher = more
diverse), the expected direction for check 6 is inverted from how the original spec
phrased it: we check whether ai/ai_edited essays show a HIGHER repeat rate, the polarity-
correct equivalent of "lower diversity."
"""
import json
import random
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F

ROOT = Path(__file__).resolve().parent.parent
ESSAYS_PATH = ROOT / "dataset" / "essays.jsonl"
SPLITS_PATH = ROOT / "dataset" / "splits.json"
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"
OUT_PATH = ROOT / "dataset" / "step3_feature_sample.jsonl"

SEED = 42
N_PER_CLASS = 10


def load_data():
    essays = [json.loads(l) for l in ESSAYS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = set(splits["train"])
    sentences = [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return essays, train_ids, sentences


def build_sample(essays, train_ids):
    rng = random.Random(SEED)
    by_label = {"human": [], "ai": [], "hybrid": []}
    for e in essays:
        if e["id"] in train_ids and not e.get("excluded"):
            by_label[e["label"]].append(e["id"])
    sample = {}
    for label, ids in by_label.items():
        ids = sorted(ids)  # sort first for reproducibility regardless of essays.jsonl order
        rng.shuffle(ids)
        sample[label] = sorted(ids[:N_PER_CLASS])
    return sample


def main():
    essays, train_ids, sentences = load_data()

    print("=" * 70)
    print("STEP 3a: FUNCTION 4 STATUS")
    print("=" * 70)
    print("Function 4 was redesigned in Step 2 (v4) from a windowed diversity ratio to a")
    print("whole-essay bigram/trigram repeat rate. Spot-tested result (from Step 2):")
    print("  h_006 (human) bigram_repeat_rate=0.0118, h_016 (human)=0.0100,")
    print("  ai_001 (ai)=0.0183, hy_050 (hybrid)=0.0421 -- real spread confirmed, no")
    print("  ceiling effect. Treating Step 3a as already satisfied; re-exercised at scale")
    print("  in Step 3c/3d below using the current essay_ngram_repetition() function.")

    print("\n" + "=" * 70)
    print("STEP 3b: STRATIFIED SAMPLE")
    print("=" * 70)
    sample = build_sample(essays, train_ids)
    print(f"Seed: {SEED}")
    for label, ids in sample.items():
        print(f"\n{label} ({len(ids)}): {ids}")

    sample_ids = set(sample["human"]) | set(sample["ai"]) | set(sample["hybrid"])

    rows_by_essay = {}
    for r in sentences:
        if r["essay_id"] in sample_ids:
            rows_by_essay.setdefault(r["essay_id"], []).append(r)
    for eid in rows_by_essay:
        rows_by_essay[eid].sort(key=lambda r: r["sentence_idx"])

    print(f"\nTotal sentences across sample: {sum(len(v) for v in rows_by_essay.values())}")

    print("\n" + "=" * 70)
    print("STEP 3c: RUNNING ALL 4 FUNCTIONS ON FULL SAMPLE")
    print("=" * 70)
    print("Loading GPT-2...")
    F.get_model()

    feature_rows = []
    for i, eid in enumerate(sorted(sample_ids), 1):
        essay_rows = rows_by_essay[eid]
        sentences_text = [r["sentence_text"] for r in essay_rows]
        print(f"[{i}/{len(sample_ids)}] {eid} ({len(sentences_text)} sentences)...", flush=True)

        perplexities = []
        logprob_vars = []
        for idx, s in enumerate(sentences_text):
            start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
            context = " ".join(sentences_text[start:idx])
            token_lps = F.sentence_token_logprobs(s, context)
            perplexities.append(F.sentence_perplexity(token_lps))
            logprob_vars.append(F.token_logprob_variance(token_lps))

        burst = F.essay_burstiness(sentences_text, perplexities)
        ngram = F.essay_ngram_repetition(sentences_text)

        for idx, r in enumerate(essay_rows):
            feature_rows.append({
                "essay_id": eid,
                "sentence_idx": r["sentence_idx"],
                "sentence_label": r["sentence_label"],
                "essay_label": r["essay_label"],
                "perplexity": perplexities[idx],
                "logprob_variance": logprob_vars[idx],
                "sentence_length_variance": burst["sentence_length_variance"],
                "syntactic_depth_variance": burst["syntactic_depth_variance"],
                "perplexity_variance": burst["perplexity_variance"],
                "bigram_repeat_rate": ngram["bigram_repeat_rate"],
                "trigram_repeat_rate": ngram["trigram_repeat_rate"],
                "ngram_repeat_rate": ngram["ngram_repeat_rate"],
                "transition_repetition_rate": ngram["transition_repetition_rate"],
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in feature_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(feature_rows)} feature rows to {OUT_PATH.relative_to(ROOT)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3d: DIRECTIONAL CHECKS")
    print("=" * 70)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    # Sentence-level checks (1 & 2): human-labeled sentences vs ai/ai_edited-labeled
    # sentences, using PER-SENTENCE labels (so hybrid essays contribute both groups).
    human_sent = [r for r in feature_rows if r["sentence_label"] == "human"]
    ai_sent = [r for r in feature_rows if r["sentence_label"] in ("ai", "ai_edited")]

    ppl_human = mean(r["perplexity"] for r in human_sent)
    ppl_ai = mean(r["perplexity"] for r in ai_sent)
    var_human = mean(r["logprob_variance"] for r in human_sent)
    var_ai = mean(r["logprob_variance"] for r in ai_sent)

    print(f"\n1. Perplexity (sentence-level, n_human={len(human_sent)}, n_ai/ai_edited={len(ai_sent)}):")
    print(f"   human mean = {ppl_human:.2f}, ai/ai_edited mean = {ppl_ai:.2f}")
    check1 = ppl_ai < ppl_human
    print(f"   Expected: ai/ai_edited LOWER. {'MATCHES' if check1 else 'DOES NOT MATCH'} "
          f"(diff = {ppl_human - ppl_ai:+.2f})")

    print(f"\n2. Log-prob variance (sentence-level):")
    print(f"   human mean = {var_human:.4f}, ai/ai_edited mean = {var_ai:.4f}")
    check2 = var_ai < var_human
    print(f"   Expected: ai/ai_edited LOWER. {'MATCHES' if check2 else 'DOES NOT MATCH'} "
          f"(diff = {var_human - var_ai:+.4f})")

    # Essay-level checks (3-7): human essays vs ai essays ONLY (hybrid essays are a blend
    # and reported separately for reference, not folded into the pass/fail comparison,
    # since essay-wide burstiness/repetition can't be meaningfully split by sentence).
    def essay_level_values(label, field):
        seen = set()
        vals = []
        for r in feature_rows:
            if r["essay_label"] == label and r["essay_id"] not in seen:
                seen.add(r["essay_id"])
                vals.append(r[field])
        return vals

    def report_essay_check(n, name, field, expect_lower_for_ai):
        h_vals = essay_level_values("human", field)
        a_vals = essay_level_values("ai", field)
        hy_vals = essay_level_values("hybrid", field)
        h_mean, a_mean, hy_mean = mean(h_vals), mean(a_vals), mean(hy_vals)
        print(f"\n{n}. {name} (essay-level, n_human=10, n_ai=10, n_hybrid=10 for reference):")
        print(f"   human mean = {h_mean:.4f}, ai mean = {a_mean:.4f}, hybrid mean (reference) = {hy_mean:.4f}")
        matches = (a_mean < h_mean) if expect_lower_for_ai else (a_mean > h_mean)
        direction = "LOWER" if expect_lower_for_ai else "HIGHER"
        print(f"   Expected: ai {direction} than human. {'MATCHES' if matches else 'DOES NOT MATCH'} "
              f"(diff = {(h_mean - a_mean) if expect_lower_for_ai else (a_mean - h_mean):+.4f})")
        return matches

    check3 = report_essay_check(3, "sentence_length_variance", "sentence_length_variance", expect_lower_for_ai=True)
    check4 = report_essay_check(4, "syntactic_depth_variance", "syntactic_depth_variance", expect_lower_for_ai=True)
    check5 = report_essay_check(5, "perplexity_variance (WATCH ITEM - reversed on single pair in Step 2)",
                                 "perplexity_variance", expect_lower_for_ai=True)
    check6 = report_essay_check(6, "ngram_repeat_rate (polarity-flipped: checking HIGHER for ai, "
                                    "the repeat-rate equivalent of 'lower diversity')",
                                 "ngram_repeat_rate", expect_lower_for_ai=False)
    check7 = report_essay_check(7, "transition_repetition_rate", "transition_repetition_rate", expect_lower_for_ai=False)

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3e: GO/NO-GO RECOMMENDATION")
    print("=" * 70)
    checks = {
        "1_perplexity": check1,
        "2_logprob_variance": check2,
        "3_sentence_length_variance": check3,
        "4_syntactic_depth_variance": check4,
        "5_perplexity_variance": check5,
        "6_ngram_repeat_rate": check6,
        "7_transition_repetition_rate": check7,
    }
    n_matching = sum(checks.values())
    print(f"\n{n_matching}/7 checks matched the expected direction:")
    for name, result in checks.items():
        print(f"  {name}: {'MATCH' if result else 'NO MATCH'}")

    if not check5:
        print("\nNote on check 5 (perplexity_variance): still reversed at 30-essay scale, "
              "same as the single-pair Step 2 result. NOT modifying the implementation per "
              "instructions. Possible cause worth considering (not yet implemented, needs "
              "approval): ai essays in this sample may have very different sentence counts "
              "than human essays (seen in Step 2: 83 vs 33 sentences for one pair), so raw "
              "per-sentence perplexity variance may be partly driven by essay length/sentence "
              "count rather than genuine burstiness - normalizing by sentence count, or using "
              "a length-controlled sub-sample of sentences per essay, could be a fix to "
              "evaluate before Step 4.")


if __name__ == "__main__":
    main()
