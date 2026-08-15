"""
Diagnostic test: check whether the ORIGINAL (pre-Phase-2-Step-4a-regeneration) hybrid
essays in class AH_original_backup/ - for tabs 33, 78, 98, 99, 100 - show better
human/ai_edited separation than the current mistral-regenerated production versions.
Does not touch any production file.
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
BACKUP_DIR = ROOT / "class AH_original_backup"
CLASS_AH_DIR = ROOT / "class AH"

TABS = [33, 78, 98, 99, 100]


def analyze(eid_label, human_text, hybrid_text):
    labels, ai_fraction, alignment_ok, note = diff_hybrid_sentences(human_text, hybrid_text)
    sentences = split_sentences(hybrid_text)
    if len(sentences) != len(labels):
        print(f"  ! length mismatch: {len(sentences)} sentences vs {len(labels)} labels - skipping")
        return None

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

    print(f"  ai_fraction={ai_fraction:.0%}, n_sentences={len(sentences)}, "
          f"n_human={len(human_devs)}, n_ai_edited={len(ai_devs)}")
    print(f"  local_perplexity_deviation |mean|: human={human_mean}, ai_edited={ai_mean}")
    if human_mean is not None and ai_mean is not None:
        direction = "ai_edited HIGHER (hoped-for)" if ai_mean > human_mean else "human HIGHER (wrong direction)"
        print(f"  -> {direction}")
    return human_mean, ai_mean


def main():
    print("=" * 70)
    print("BACKUP (pre-regeneration) HYBRIDS vs PRODUCTION (mistral) HYBRIDS")
    print("=" * 70)
    F.get_model()

    for tab in TABS:
        eid = f"h_{tab:03d}"
        human_path = CLASS_H_DIR / f"Tab_{tab}.txt"
        backup_path = BACKUP_DIR / f"Tab_{tab}_hybrid.txt"
        prod_path = CLASS_AH_DIR / f"Tab_{tab}_hybrid.txt"

        human_text = human_path.read_text(encoding="utf-8", errors="replace").strip()
        backup_text = backup_path.read_text(encoding="utf-8", errors="replace").strip()
        prod_text = prod_path.read_text(encoding="utf-8", errors="replace").strip()

        print(f"\n[{eid}] BACKUP (pre-regeneration, possibly manually GPT-created):")
        analyze(eid, human_text, backup_text)

        print(f"[{eid}] PRODUCTION (mistral-regenerated, voice-matched):")
        analyze(eid, human_text, prod_text)


if __name__ == "__main__":
    main()
