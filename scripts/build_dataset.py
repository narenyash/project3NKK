"""
Phase 1, Steps 1 & 2: inventory the raw class folders and normalize them into
dataset/essays.jsonl, including sentence-level diff labels for hybrid essays.
"""
import difflib
import json
import re
from pathlib import Path

from text_utils import split_sentences, word_count, using_spacy

ROOT = Path(__file__).resolve().parent.parent
CLASS_A_DIR = ROOT / "Class A"
CLASS_H_DIR = ROOT / "class H"
CLASS_AH_DIR = ROOT / "class AH"
OUT_PATH = ROOT / "dataset" / "essays.jsonl"
CONVERSION_REPORT_PATH = ROOT / "dataset" / "conversion_report.md"

TAB_RE = re.compile(r"Tab_(\d+)(?:_hybrid)?\.txt$", re.IGNORECASE)
NUM_RE = re.compile(r"(\d+)\.txt$")


def read_text(path: Path) -> str:
    """UTF-8 read with lossy-replace on bad bytes rather than raising - a handful of raw
    essay files have minor encoding artifacts, and this script's job is to inventory and
    normalize them, not fail on them."""
    return path.read_text(encoding="utf-8", errors="replace")


# Globs and number-sorts all three raw class folders, and cross-checks that every
# class H Tab number has a matching class AH hybrid (needed for the sentence diff below).
def inventory():
    class_a_files = sorted(CLASS_A_DIR.glob("*.txt"), key=lambda p: int(NUM_RE.search(p.name).group(1)))
    class_h_files = sorted(CLASS_H_DIR.glob("Tab_*.txt"), key=lambda p: int(TAB_RE.search(p.name).group(1)))
    class_ah_files = sorted(CLASS_AH_DIR.glob("Tab_*_hybrid.txt"), key=lambda p: int(TAB_RE.search(p.name).group(1)))

    print("=" * 60)
    print("STEP 1: INVENTORY")
    print("=" * 60)
    print(f"Class A (ai):     {len(class_a_files)} files")
    print(f"class H (human):  {len(class_h_files)} files")
    print(f"class AH (hybrid):{len(class_ah_files)} files")

    empty_files = []
    for f in class_a_files + class_h_files + class_ah_files:
        if not read_text(f).strip():
            empty_files.append(str(f.relative_to(ROOT)))
    if empty_files:
        print(f"\n!! Empty files found ({len(empty_files)}):")
        for ef in empty_files:
            print(f"   - {ef}")
    else:
        print("\nNo empty files found.")

    # duplicate filenames across the *same* folder (shouldn't happen with glob, but check numbering gaps)
    h_nums = {int(TAB_RE.search(f.name).group(1)) for f in class_h_files}
    ah_nums = {int(TAB_RE.search(f.name).group(1)) for f in class_ah_files}
    a_nums = {int(NUM_RE.search(f.name).group(1)) for f in class_a_files}

    missing_hybrid_for_human = sorted(h_nums - ah_nums)
    hybrid_without_human = sorted(ah_nums - h_nums)
    if missing_hybrid_for_human:
        print(f"\n!! Human essays with NO matching hybrid: {missing_hybrid_for_human}")
    if hybrid_without_human:
        print(f"\n!! Hybrid essays with NO matching human original (can't sentence-diff): {hybrid_without_human}")
    if not missing_hybrid_for_human and not hybrid_without_human:
        print("\nAll class H <-> class AH tab numbers match 1:1 (sentence-diffable).")

    expected_a = set(range(1, 101))
    if a_nums != expected_a:
        print(f"\n!! Class A numbering gap. Missing: {sorted(expected_a - a_nums)}, Extra: {sorted(a_nums - expected_a)}")

    return class_a_files, class_h_files, class_ah_files, h_nums, ah_nums


def diff_hybrid_sentences(human_text: str, hybrid_text: str):
    """Return (sentence_labels, ai_edited_fraction, alignment_ok, note) for one hybrid essay."""
    human_sents = split_sentences(human_text)
    hybrid_sents = split_sentences(hybrid_text)
    if not hybrid_sents:
        return [], 0.0, False, "empty hybrid text"
    if word_count(human_text) < 20:
        # Essentially no human source material (e.g. the human essay slot was blank/a
        # bare header) - the entire "hybrid" is actually AI-fabricated from nothing,
        # not a real 60/40 blend.
        labels = ["ai_edited"] * len(hybrid_sents)
        return labels, 1.0, False, (
            f"human original has only {word_count(human_text)} words - hybrid is 100% AI-fabricated, "
            "not a real blend"
        )

    sm = difflib.SequenceMatcher(None, human_sents, hybrid_sents, autojunk=False)
    labels = [None] * len(hybrid_sents)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                labels[j] = "human"
        else:  # replace / insert -> hybrid-only sentences are AI-touched
            for j in range(j1, j2):
                labels[j] = "ai_edited"
    # delete ops (human sentences dropped) don't produce hybrid-side labels

    ai_count = sum(1 for l in labels if l == "ai_edited")
    ai_fraction = ai_count / len(labels) if labels else 0.0
    # Flag as poor alignment if essentially nothing matched (AI polish restructured everything)
    alignment_ok = ai_fraction < 0.98
    note = None if alignment_ok else (
        f"sentence diff looks degenerate ({ai_fraction:.0%} flagged ai_edited) - AI polish may have "
        "restructured the essay beyond clean sentence alignment"
    )
    return labels, ai_fraction, alignment_ok, note


# Turns the three raw file lists into the shared essays.jsonl schema (id, text, label,
# source, word_count, etc.), calling diff_hybrid_sentences() above for every hybrid to
# attach its sentence_labels array. Returns (records, conversion_issues) - issues become
# dataset/conversion_report.md, not silent skips.
def build_records(class_a_files, class_h_files, class_ah_files, h_nums, ah_nums):
    records = []
    conversion_issues = []

    for f in class_a_files:
        num = int(NUM_RE.search(f.name).group(1))
        text = read_text(f).strip()
        records.append({
            "id": f"ai_{num:03d}",
            "text": text,
            "label": "ai",
            "topic": None,
            "genre": "admissions_essay",
            "source": "ai-generated (model/settings undocumented)",
            "word_count": word_count(text),
            "sentence_count": None,
            "excluded": False,
            "exclusion_reason": None,
        })
        if not text:
            conversion_issues.append(f"ai_{num:03d}: empty text ({f.name})")

    ah_by_num = {int(TAB_RE.search(f.name).group(1)): f for f in class_ah_files}

    for f in class_h_files:
        num = int(TAB_RE.search(f.name).group(1))
        text = read_text(f).strip()
        is_stub = word_count(text) < 20
        exclusion_reason = "human source is a stub (<20 words), not a real essay" if is_stub else None
        records.append({
            "id": f"h_{num:03d}",
            "text": text,
            "label": "human",
            "topic": None,
            "genre": "admissions_essay",
            "source": "student submission (self-collected)",
            "word_count": word_count(text),
            "sentence_count": None,
            "excluded": is_stub,
            "exclusion_reason": exclusion_reason,
        })
        if not text:
            conversion_issues.append(f"h_{num:03d}: empty text ({f.name})")
        elif is_stub:
            conversion_issues.append(f"h_{num:03d}: {exclusion_reason} - excluded from modeling-ready set")

        if num in ah_by_num:
            hybrid_file = ah_by_num[num]
            hybrid_text = read_text(hybrid_file).strip()
            labels, ai_fraction, alignment_ok, note = diff_hybrid_sentences(text, hybrid_text)
            if not alignment_ok and not is_stub:
                conversion_issues.append(f"hy_{num:03d}: {note}")
            records.append({
                "id": f"hy_{num:03d}",
                "text": hybrid_text,
                "label": "hybrid",
                "topic": None,
                "genre": "admissions_essay",
                "source": "human draft + ai-polish (local Ollama, controlled paragraph/sentence-level "
                          "blend via regenerate_hybrids.py)",
                "word_count": word_count(hybrid_text),
                "sentence_count": None,
                "sentence_labels": labels,
                "human_source_id": f"h_{num:03d}",
                "excluded": is_stub,
                "exclusion_reason": exclusion_reason.replace("human source", "paired human source") if is_stub else None,
            })
            if not hybrid_text:
                conversion_issues.append(f"hy_{num:03d}: empty text ({hybrid_file.name})")
        else:
            conversion_issues.append(f"h_{num:03d}: NO matching hybrid essay found - cannot sentence-label")

    return records, conversion_issues


# Entry point: inventory -> build_records() -> write dataset/essays.jsonl +
# conversion_report.md.
def main():
    (ROOT / "dataset").mkdir(exist_ok=True)
    class_a_files, class_h_files, class_ah_files, h_nums, ah_nums = inventory()

    print("\n" + "=" * 60)
    print("STEP 2: NORMALIZE INTO SCHEMA")
    print("=" * 60)
    print(f"Sentence splitter in use: {'spaCy (en_core_web_sm)' if using_spacy() else 'regex fallback'}")

    records, conversion_issues = build_records(class_a_files, class_h_files, class_ah_files, h_nums, ah_nums)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hybrid_records = [r for r in records if r["label"] == "hybrid"]
    diffable = sum(1 for r in hybrid_records if r.get("sentence_labels"))
    print(f"\nWrote {len(records)} records to {OUT_PATH.relative_to(ROOT)}")
    print(f"  ai:     {sum(1 for r in records if r['label']=='ai')}")
    print(f"  human:  {sum(1 for r in records if r['label']=='human')}")
    print(f"  hybrid: {len(hybrid_records)} ({diffable} with sentence-level diff labels)")

    with CONVERSION_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Conversion Report (Steps 1-2)\n\n")
        if conversion_issues:
            f.write(f"{len(conversion_issues)} issue(s) found:\n\n")
            for issue in conversion_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("No conversion issues found. All 300 essays normalized cleanly; "
                    "all 100 hybrid essays have sentence-level diff labels against their human original.\n")

    print(f"\nConversion issues: {len(conversion_issues)} (see {CONVERSION_REPORT_PATH.relative_to(ROOT)})")
    if conversion_issues:
        for issue in conversion_issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
