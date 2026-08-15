"""
Analyze the diagnostic experiment_hybrids/ (no voice-matching instruction) for
human/ai_edited separation, compared against the current production versions of the
same essays.
"""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import features as F
from build_dataset import diff_hybrid_sentences
from text_utils import split_sentences

ROOT = Path(__file__).resolve().parent.parent
CLASS_H_DIR = ROOT / "class H"
CLASS_AH_DIR = ROOT / "class AH"
EXP_DIR = ROOT / "experiment_hybrids"

TABS = [1, 6, 16, 45, 50, 56, 57, 100]


def analyze(human_text, hybrid_text):
    labels, ai_fraction, alignment_ok, note = diff_hybrid_sentences(human_text, hybrid_text)
    sentences = split_sentences(hybrid_text)
    if len(sentences) != len(labels):
        return None, None, ai_fraction

    perplexities = []
    for idx, s in enumerate(sentences):
        start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
        context = " ".join(sentences[start:idx])
        token_lps = F.sentence_token_logprobs(s, context)
        perplexities.append(F.sentence_perplexity(token_lps))

    local_dev = [F.local_perplexity_deviation(perplexities, i) for i in range(len(sentences))]
    human_devs = [abs(d) for d, l in zip(local_dev, labels) if l == "human" and d is not None]
    ai_devs = [abs(d) for d, l in zip(local_dev, labels) if l == "ai_edited" and d is not None]
    human_mean = statistics.mean(human_devs) if human_devs else None
    ai_mean = statistics.mean(ai_devs) if ai_devs else None
    return human_mean, ai_mean, ai_fraction


def main():
    print("=" * 70)
    print("EXPERIMENTAL (no voice-match) vs PRODUCTION (voice-matched) SEPARATION")
    print("=" * 70)
    F.get_model()

    exp_human_all, exp_ai_all = [], []
    prod_human_all, prod_ai_all = [], []

    for tab in TABS:
        eid = f"h_{tab:03d}"
        human_text = (CLASS_H_DIR / f"Tab_{tab}.txt").read_text(encoding="utf-8", errors="replace").strip()
        exp_text = (EXP_DIR / f"{eid}_exp_hybrid.txt").read_text(encoding="utf-8", errors="replace").strip()
        prod_text = (CLASS_AH_DIR / f"Tab_{tab}_hybrid.txt").read_text(encoding="utf-8", errors="replace").strip()

        eh, ea, ef = analyze(human_text, exp_text)
        ph, pa, pf = analyze(human_text, prod_text)

        print(f"\n[{eid}]")
        print(f"  EXPERIMENTAL (no voice-match): human={eh}, ai_edited={ea} (ai_fraction={ef:.0%})")
        print(f"  PRODUCTION   (voice-matched):  human={ph}, ai_edited={pa} (ai_fraction={pf:.0%})")

        if eh is not None and ea is not None:
            exp_human_all.append(eh)
            exp_ai_all.append(ea)
        if ph is not None and pa is not None:
            prod_human_all.append(ph)
            prod_ai_all.append(pa)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean of per-essay means)")
    print("=" * 70)
    print(f"EXPERIMENTAL: human={statistics.mean(exp_human_all):.4f}, ai_edited={statistics.mean(exp_ai_all):.4f}")
    print(f"PRODUCTION:   human={statistics.mean(prod_human_all):.4f}, ai_edited={statistics.mean(prod_ai_all):.4f}")


if __name__ == "__main__":
    main()
