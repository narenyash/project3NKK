"""
Phase 4, Step 1: prepare test-split data. This is the FIRST time test-split essays have
their features extracted (they were already sentence-segmented with all fixes applied,
since text_utils.split_sentences() and build_sentences.py process all essays uniformly
regardless of split - but GPT-2/spaCy feature extraction was always filtered to
split=="train" only, until now).

Reports fix-impact counts specifically for the test split (previously only reported for
train+test combined or train-only), extracts the 7 raw features, and z-scores/clips using
the EXISTING, unmodified corpus_baseline.json - never recomputed here.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F
from build_sentences import BARE_FRAGMENT_RE, CITATION_APOSTROPHE_RE
from text_utils import split_sentences

ROOT = Path(__file__).resolve().parent.parent
ESSAYS_PATH = ROOT / "dataset" / "essays.jsonl"
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"
BASELINE_PATH = ROOT / "dataset" / "corpus_baseline.json"
OUT_PATH = ROOT / "dataset" / "features_test_zscored.jsonl"

FEATURES = ["perplexity", "logprob_variance", "sentence_length_variance",
            "syntactic_depth_variance", "perplexity_variance", "ngram_repeat_rate",
            "transition_repetition_rate"]
CLIP_LO, CLIP_HI = -5.0, 5.0
TAB_HEADER_RE = re.compile(r"^Tab \d+")


def main():
    essays = [json.loads(l) for l in ESSAYS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    sentences = [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_sentence_rows = [r for r in sentences if r["split"] == "test"]
    test_essay_ids = sorted(set(r["essay_id"] for r in test_sentence_rows))
    by_id = {e["id"]: e for e in essays}

    print("=" * 70)
    print("PHASE 4 STEP 1: PREPARE TEST-SPLIT DATA")
    print("=" * 70)
    print(f"Test-split essays: {len(test_essay_ids)}")
    print(f"Test-split sentence rows (already segmented with all fixes, from sentences.jsonl): {len(test_sentence_rows)}")

    # --- Fix-impact breakdown, test split specifically ---
    print("\n--- Fix-impact counts, TEST SPLIT ONLY ---")

    tab_n_affected = [eid for eid in test_essay_ids if TAB_HEADER_RE.match(by_id[eid]["text"])]
    print(f"Tab-N header fix: {len(tab_n_affected)}/{len(test_essay_ids)} test essays affected: {tab_n_affected}")

    bare_fragment_count = 0
    bare_fragment_essays = []
    for eid in test_essay_ids:
        raw_sentences = split_sentences(by_id[eid]["text"])
        for s in raw_sentences:
            st = s.strip()
            if BARE_FRAGMENT_RE.match(st) or CITATION_APOSTROPHE_RE.match(st):
                bare_fragment_count += 1
                bare_fragment_essays.append((eid, st))
    print(f"Bare-fragment filter: {bare_fragment_count} fragment(s) found in test essays "
          f"(should be 0 - they'd already be excluded from sentences.jsonl if present, "
          f"this re-check confirms none remain unfiltered): {bare_fragment_essays}")

    print("Apostrophe mis-split fix: only 3 known instances existed corpus-wide "
          f"(h_025, h_087, hy_087) - checking split membership: "
          f"h_025 in test={'h_025' in test_essay_ids}, h_087 in test={'h_087' in test_essay_ids}, "
          f"hy_087 in test={'hy_087' in test_essay_ids}")

    # --- Feature extraction ---
    print("\n" + "=" * 70)
    print("EXTRACTING 7 RAW FEATURES (test split, first time)")
    print("=" * 70)
    print("Loading GPT-2...")
    F.get_model()

    by_essay = {}
    for r in test_sentence_rows:
        by_essay.setdefault(r["essay_id"], []).append(r)
    for eid in by_essay:
        by_essay[eid].sort(key=lambda r: r["sentence_idx"])

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    feature_rows = []
    for i, eid in enumerate(sorted(by_essay.keys()), 1):
        essay_rows = by_essay[eid]
        sentences_text = [r["sentence_text"] for r in essay_rows]

        perplexities, logprob_vars = [], []
        for idx, s in enumerate(sentences_text):
            start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
            context = " ".join(sentences_text[start:idx])
            token_lps = F.sentence_token_logprobs(s, context)
            perplexities.append(F.sentence_perplexity(token_lps))
            logprob_vars.append(F.token_logprob_variance(token_lps))

        burst = F.essay_burstiness(sentences_text, perplexities)
        ngram = F.essay_ngram_repetition(sentences_text)

        for idx, r in enumerate(essay_rows):
            raw = {
                "essay_id": eid,
                "sentence_idx": r["sentence_idx"],
                "sentence_label": r["sentence_label"],
                "essay_label": r["essay_label"],
                "topic": r["topic"],
                "perplexity": perplexities[idx],
                "logprob_variance": logprob_vars[idx],
                "sentence_length_variance": burst["sentence_length_variance"],
                "syntactic_depth_variance": burst["syntactic_depth_variance"],
                "perplexity_variance": burst["perplexity_variance"],
                "ngram_repeat_rate": ngram["ngram_repeat_rate"],
                "transition_repetition_rate": ngram["transition_repetition_rate"],
            }
            # z-score + clip using EXISTING baseline
            for field in FEATURES:
                v = raw[field]
                m, s = baseline[field]["mean"], baseline[field]["std"]
                if v is None or s == 0:
                    raw[f"{field}_z"] = None
                    continue
                z = (v - m) / s
                z = max(CLIP_LO, min(CLIP_HI, z))
                raw[f"{field}_z"] = z
            feature_rows.append(raw)

        if i % 15 == 0 or i == len(by_essay):
            print(f"[{i}/{len(by_essay)}] essays done", flush=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in feature_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(feature_rows)} rows to {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
