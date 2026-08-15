"""
Phase 2, Step 7.5a: spot-test the two new local features on 1 human essay + 1 hybrid
essay before running at scale.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "dataset" / "features_train_raw.jsonl"
SENTENCES_PATH = ROOT / "dataset" / "sentences.jsonl"


def load_essay(eid):
    feat_rows = [json.loads(l) for l in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    sent_rows = [json.loads(l) for l in SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    fr = sorted([r for r in feat_rows if r["essay_id"] == eid], key=lambda r: r["sentence_idx"])
    sr = sorted([r for r in sent_rows if r["essay_id"] == eid], key=lambda r: r["sentence_idx"])
    return fr, sr


def spot_test(eid):
    fr, sr = load_essay(eid)
    sentences_text = [r["sentence_text"] for r in sr]
    perplexities = [r["perplexity"] for r in fr]
    labels = [r["sentence_label"] for r in sr]

    local_ppl_dev = [F.local_perplexity_deviation(perplexities, i) for i in range(len(sentences_text))]
    burst = F.local_burstiness_disruption(sentences_text)

    print(f"\n[{eid}] ({len(sentences_text)} sentences)")
    print(f"{'idx':>4s} {'label':>10s} {'ppl_dev':>10s} {'len_dev':>10s} {'depth_dev':>10s}  text")
    for i in range(len(sentences_text)):
        ppl_d = local_ppl_dev[i]
        len_d = burst["local_length_deviation"][i]
        depth_d = burst["local_depth_deviation"][i]
        fmt = lambda v: f"{v:10.3f}" if v is not None else f"{'None':>10s}"
        print(f"{i:4d} {labels[i]:>10s} {fmt(ppl_d)} {fmt(len_d)} {fmt(depth_d)}  {sentences_text[i][:70]!r}")

    return local_ppl_dev, burst, labels


def main():
    print("=" * 70)
    print("STEP 7.5a: SPOT-TEST LOCAL FEATURES")
    print("=" * 70)

    spot_test("h_006")  # human
    ppl_dev, burst, labels = spot_test("hy_050")  # hybrid, known ai_edited sentences

    print("\n--- Does ai_edited stand out from human within hy_050? ---")
    import statistics
    for name, vals in [("local_perplexity_deviation", ppl_dev),
                        ("local_length_deviation", burst["local_length_deviation"]),
                        ("local_depth_deviation", burst["local_depth_deviation"])]:
        human_vals = [abs(v) for v, l in zip(vals, labels) if l == "human" and v is not None]
        ai_vals = [abs(v) for v, l in zip(vals, labels) if l == "ai_edited" and v is not None]
        h_mean = statistics.mean(human_vals) if human_vals else None
        a_mean = statistics.mean(ai_vals) if ai_vals else None
        print(f"{name}: human |mean|={h_mean}, ai_edited |mean|={a_mean}")


if __name__ == "__main__":
    main()
