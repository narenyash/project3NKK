"""
Fix for Phase 1 limitation #2: the original hybrid essays were produced by asking
Ollama to rewrite the *whole* essay while "keeping ~60% unchanged" - in practice the
model frequently rewrote every sentence, missing that target entirely (51/100 hybrids
had 0% verbatim retention).

This script replaces that with a controlled blend: split each essay into paragraphs
(or sentences, if the essay lacks real paragraph breaks), deterministically select
~40% of those units to send to Ollama for enhancement, and leave the rest byte-for-byte
untouched. This guarantees ~60% verbatim retention by construction instead of hoping
a whole-essay rewrite happens to preserve it.

Tab_21 and Tab_26 are skipped: their human "source" is a 2-word stub, not a real essay,
so there's nothing genuine to blend from (handled as an exclusion downstream in
build_dataset.py / make_split.py).
"""
import random
import re
import shutil
from math import ceil
from pathlib import Path

import ollama

from text_utils import split_sentences, word_count

ROOT = Path(__file__).resolve().parent.parent
CLASS_H_DIR = ROOT / "class H"
CLASS_AH_DIR = ROOT / "class AH"
BACKUP_DIR = ROOT / "class AH_original_backup"

TAB_RE = re.compile(r"Tab_(\d+)\.txt$", re.IGNORECASE)
SKIP_TABS = {21, 26}
TARGET_AI_FRACTION = 0.4
SEED = 42
MODEL = "mistral:latest"

ENHANCE_PROMPT = """Rewrite the following paragraph from a college admissions essay to be more polished, \
vivid, and detailed, while keeping the same core content, meaning, and voice. Keep it roughly the same \
length. Return ONLY the rewritten paragraph, no preamble or commentary.

Paragraph:
---
{unit}
---"""


def get_units(text: str):
    """Paragraph-level units if the essay has real paragraph breaks, else sentence-level."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs, "paragraph"
    return split_sentences(text), "sentence"


# One paragraph/sentence, AI-polished via Ollama. Falls back to the original unit
# verbatim on any error - a failed Ollama call should degrade to "less AI content than
# intended," never crash the whole run or fabricate placeholder text.
def enhance_unit(unit: str) -> str:
    try:
        response = ollama.generate(model=MODEL, prompt=ENHANCE_PROMPT.format(unit=unit), stream=False)
        result = response["response"].strip()
        return result if result else unit
    except Exception as e:
        print(f"    ! Ollama error, keeping unit verbatim: {e}")
        return unit


# One essay -> one hybrid: splits into units (get_units), deterministically (seeded rng)
# selects ~TARGET_AI_FRACTION of them by index, sends only those through enhance_unit(),
# and rejoins everything (untouched units included) in original order. Returns
# (hybrid_text, unit_kind, k, n) so the caller can log exactly how many units of what
# kind were selected, for the retention-rate reporting in main() below.
def build_hybrid(text: str, rng: random.Random):
    units, unit_kind = get_units(text)
    n = len(units)
    k = max(1, min(n - 1, ceil(TARGET_AI_FRACTION * n)))
    indices = list(range(n))
    rng.shuffle(indices)
    ai_indices = set(indices[:k])

    new_units = []
    for i, unit in enumerate(units):
        if i in ai_indices:
            new_units.append(enhance_unit(unit))
        else:
            new_units.append(unit)

    joiner = "\n\n" if unit_kind == "paragraph" else " "
    return joiner.join(new_units), unit_kind, k, n


# Entry point: backs up the existing class AH/ once (never overwrites an existing
# backup), then regenerates every valid hybrid via build_hybrid() and overwrites
# class AH/ in place. Skips Tab_21/Tab_26 (SKIP_TABS - no valid human source to blend
# from) and prints per-essay + summary retention-rate stats.
def main():
    if BACKUP_DIR.exists():
        print(f"{BACKUP_DIR.relative_to(ROOT)} already exists - not overwriting backup. "
              "Delete it manually first if you want a fresh backup.")
    else:
        shutil.copytree(CLASS_AH_DIR, BACKUP_DIR)
        backup_count = len(list(BACKUP_DIR.glob("*.txt")))
        print(f"Backed up {backup_count} original hybrid files to {BACKUP_DIR.relative_to(ROOT)}")

    human_files = sorted(CLASS_H_DIR.glob("Tab_*.txt"), key=lambda p: int(TAB_RE.search(p.name).group(1)))
    rng = random.Random(SEED)

    regenerated = 0
    skipped = []
    for f in human_files:
        num = int(TAB_RE.search(f.name).group(1))
        if num in SKIP_TABS:
            skipped.append(num)
            continue

        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if word_count(text) < 20:
            skipped.append(num)
            continue

        print(f"Regenerating Tab_{num}_hybrid.txt...", end=" ", flush=True)
        hybrid_text, unit_kind, k, n = build_hybrid(text, rng)
        out_path = CLASS_AH_DIR / f"Tab_{num}_hybrid.txt"
        out_path.write_text(hybrid_text, encoding="utf-8")
        print(f"done ({unit_kind}-level, {k}/{n} units AI-enhanced)")
        regenerated += 1

    print(f"\nRegenerated {regenerated} hybrid essays.")
    print(f"Skipped (no valid human source): {sorted(skipped)}")


if __name__ == "__main__":
    main()
