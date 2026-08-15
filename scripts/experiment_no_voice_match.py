"""
Diagnostic experiment (not part of the production pipeline): regenerate a small sample
of hybrid essays WITHOUT the "keep the same voice" instruction used in
regenerate_hybrids.py, to test whether that instruction is suppressing detectability of
AI-edited sentences. Writes to experiment_hybrids/ - does NOT touch class AH/ or any
production dataset file.
"""
import json
import random
import sys
from math import ceil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import ollama

from build_dataset import diff_hybrid_sentences
from text_utils import split_sentences, word_count

ROOT = Path(__file__).resolve().parent.parent
CLASS_H_DIR = ROOT / "class H"
OUT_DIR = ROOT / "experiment_hybrids"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_TABS = [1, 6, 16, 45, 50, 56, 57, 100]
TARGET_AI_FRACTION = 0.4
SEED = 42
MODEL = "mistral:latest"

# The ONLY change from regenerate_hybrids.py's ENHANCE_PROMPT: no voice/length-matching
# instruction. Everything else (selection mechanism, fraction, model) is identical, so
# this isolates the effect of that one instruction.
NO_VOICE_MATCH_PROMPT = """Rewrite the following paragraph from a college admissions essay to be more polished \
and detailed. Return ONLY the rewritten paragraph, no preamble or commentary.

Paragraph:
---
{unit}
---"""


def get_units(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs, "paragraph"
    return split_sentences(text), "sentence"


def enhance_unit(unit):
    try:
        response = ollama.generate(model=MODEL, prompt=NO_VOICE_MATCH_PROMPT.format(unit=unit), stream=False)
        result = response["response"].strip()
        return result if result else unit
    except Exception as e:
        print(f"    ! Ollama error, keeping unit verbatim: {e}")
        return unit


def build_experimental_hybrid(text, rng):
    units, unit_kind = get_units(text)
    n = len(units)
    k = max(1, min(n - 1, ceil(TARGET_AI_FRACTION * n)))
    indices = list(range(n))
    rng.shuffle(indices)
    ai_indices = set(indices[:k])
    new_units = [enhance_unit(u) if i in ai_indices else u for i, u in enumerate(units)]
    joiner = "\n\n" if unit_kind == "paragraph" else " "
    return joiner.join(new_units)


def main():
    rng = random.Random(SEED)
    print("=" * 70)
    print("DIAGNOSTIC EXPERIMENT: hybrids without voice-matching instruction")
    print("=" * 70)

    for tab in SAMPLE_TABS:
        eid = f"h_{tab:03d}"
        src = CLASS_H_DIR / f"Tab_{tab}.txt"
        text = src.read_text(encoding="utf-8", errors="replace").strip()
        print(f"Generating experimental hybrid for {eid}...", end=" ", flush=True)
        hybrid_text = build_experimental_hybrid(text, rng)
        out_path = OUT_DIR / f"{eid}_exp_hybrid.txt"
        out_path.write_text(hybrid_text, encoding="utf-8")

        labels, ai_fraction, alignment_ok, note = diff_hybrid_sentences(text, hybrid_text)
        print(f"done ({ai_fraction:.0%} sentences ai_edited)")

    print(f"\nWrote {len(SAMPLE_TABS)} experimental hybrids to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
