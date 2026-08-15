"""
Blast-radius report for the "Tab N" header fix: targeted before/after perplexity
comparison for sentence 0 of the 119 train-split affected essays, plus an approximated
baseline-shift estimate. Does NOT re-run the full 13,811-sentence extraction - this is a
deliberately targeted, cheap recomputation to inform the go/no-go decision on a full
Step 4/5/7 re-run.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F
from text_utils import split_sentences

ROOT = Path(__file__).resolve().parent.parent


def main():
    essays = [json.loads(l) for l in (ROOT / "dataset" / "essays.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    splits = json.loads((ROOT / "dataset" / "splits.json").read_text(encoding="utf-8"))
    train_ids = set(splits["train"])
    old_features = [json.loads(l) for l in (ROOT / "dataset" / "features_train_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    old_sent0 = {r["essay_id"]: r["perplexity"] for r in old_features if r["sentence_idx"] == 0}

    by_id = {e["id"]: e for e in essays}

    # Re-identify the 119 affected essays using the OLD sentence-0 text (from features_train_raw's
    # source, i.e. the pre-fix sentences.jsonl) - but simplest robust way: any train essay whose
    # OLD sentence 0 perplexity we have, and whose raw text (still containing "Tab N") starts
    # with the artifact pattern.
    affected = []
    for eid in train_ids:
        e = by_id.get(eid)
        if e is None or e.get("excluded"):
            continue
        if e["text"].split("\n", 1)[0].strip().split(" ")[0] == "Tab" and e["text"].split("\n")[0].strip().split(" ")[-1].isdigit():
            affected.append(eid)

    print("=" * 70)
    print("STEP C.1: TARGETED SENTENCE-0 PERPLEXITY, BEFORE vs AFTER")
    print("=" * 70)
    print(f"Affected train-split essays re-identified: {len(affected)}")

    print("Loading GPT-2...")
    F.get_model()

    changes = []
    for eid in sorted(affected):
        e = by_id[eid]
        new_sentences = split_sentences(e["text"])
        if not new_sentences:
            continue
        new_sent0_text = new_sentences[0]
        token_lps = F.sentence_token_logprobs(new_sent0_text, "")
        new_ppl = F.sentence_perplexity(token_lps)
        old_ppl = old_sent0.get(eid)
        if old_ppl is None or new_ppl is None:
            continue
        changes.append((eid, old_ppl, new_ppl))

    print(f"\nComputed before/after for {len(changes)} essays.\n")
    diffs = [old - new for _, old, new in changes]
    ratios = [old / new if new > 0 else None for _, old, new in changes]
    print(f"{'essay_id':10s} {'old_ppl':>12s} {'new_ppl':>12s} {'ratio':>8s}")
    for eid, old, new in sorted(changes, key=lambda x: -(x[1] - x[2]))[:20]:
        ratio = old / new if new > 0 else float("inf")
        print(f"{eid:10s} {old:12.2f} {new:12.2f} {ratio:8.2f}x")
    print("  ... (showing top 20 by absolute drop)")

    print(f"\nSummary of change (old_ppl - new_ppl), n={len(diffs)}:")
    print(f"  mean drop = {statistics.mean(diffs):.2f}")
    print(f"  median drop = {statistics.median(diffs):.2f}")
    print(f"  max drop = {max(diffs):.2f}  (essay: {changes[diffs.index(max(diffs))][0]})")
    print(f"  min drop = {min(diffs):.2f}")
    n_large_drop = sum(1 for d in diffs if d > 100)
    print(f"  essays with drop > 100 perplexity points: {n_large_drop}/{len(diffs)}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP C.2: APPROXIMATED HUMAN BASELINE SHIFT (perplexity)")
    print("=" * 70)
    print("Approximation method: take the existing full-scale features_train_raw.jsonl "
          "sentence-level perplexity values, SWAP IN the corrected sentence-0 values for "
          "the affected essays only, recompute the human baseline mean/std. This is a "
          "targeted patch for blast-radius estimation, not a full re-extraction (which "
          "would also need to redo logprob_variance and word-count-dependent features).")

    old_human_all = [r["perplexity"] for r in old_features if r["sentence_label"] == "human" and r["perplexity"] is not None]
    old_mean, old_std = statistics.mean(old_human_all), statistics.pstdev(old_human_all)

    change_by_eid = {eid: new for eid, old, new in changes}
    label_by_eid = {r["essay_id"]: r["essay_label"] for r in old_features}
    new_human_all = []
    for r in old_features:
        if r["sentence_label"] != "human" or r["perplexity"] is None:
            continue
        if r["sentence_idx"] == 0 and r["essay_id"] in change_by_eid:
            new_human_all.append(change_by_eid[r["essay_id"]])
        else:
            new_human_all.append(r["perplexity"])
    new_mean, new_std = statistics.mean(new_human_all), statistics.pstdev(new_human_all)

    print(f"\nOLD human perplexity baseline: mean={old_mean:.4f}, std={old_std:.4f} (n={len(old_human_all)})")
    print(f"NEW human perplexity baseline (approx): mean={new_mean:.4f}, std={new_std:.4f} (n={len(new_human_all)})")
    print(f"Mean shift: {new_mean - old_mean:+.4f} ({(new_mean - old_mean) / old_mean:+.2%} relative)")
    print(f"Std shift: {new_std - old_std:+.4f} ({(new_std - old_std) / old_std:+.2%} relative)")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP C.3: COEFFICIENT SENSITIVITY ASSESSMENT (not retraining)")
    print("=" * 70)
    ai_all = [r["perplexity"] for r in old_features if r["sentence_label"] == "ai" and r["perplexity"] is not None]
    ai_mean = statistics.mean(ai_all)
    old_gap = old_mean - ai_mean
    new_gap = new_mean - ai_mean
    print(f"ai perplexity mean (unaffected by this fix): {ai_mean:.4f}")
    print(f"human-ai gap OLD: {old_gap:.4f}")
    print(f"human-ai gap NEW (approx): {new_gap:.4f}")
    print(f"Gap change: {new_gap - old_gap:+.4f} ({(new_gap - old_gap) / old_gap:+.2%} relative)")

    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP C.4: LABEL SKEW (already established, restated for the record)")
    print("=" * 70)
    print("ai: 0/100 affected (0%) | human: 95/98 affected (97%) | hybrid: 54/98 affected (55%)")
    print("This is a one-directional bias hitting human (and to a lesser extent hybrid) "
          "essays almost exclusively, never ai essays.")


if __name__ == "__main__":
    main()
