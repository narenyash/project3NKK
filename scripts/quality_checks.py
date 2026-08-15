"""
Phase 1, Steps 4 & 5: data quality pass (duplicates, outliers, encoding, mislabel
heuristics) and balance/diversity report. Findings are reported for manual review,
not silently auto-fixed (except trivial whitespace normalization).
"""
import hashlib
import json
import re
import statistics
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "essays.jsonl"
QUALITY_REPORT_PATH = ROOT / "dataset" / "quality_report.md"

WORD_COUNT_MIN = 250
WORD_COUNT_MAX = 1000
NEAR_DUP_THRESHOLD = 0.85

MOJIBAKE_RE = re.compile(r"[�]|Ã[\x80-\xbf]|â€[\x80-\x9f]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def load_records():
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


# Only called after the (rare) auto-fixable whitespace fix below - most checks in this
# file are read-only/report-only and never call this.
def save_records(records):
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Flags essays under 50 chars as effectively empty/stub content.
def check_empty(records):
    return [r["id"] for r in records if len(r["text"].strip()) < 50]


# Byte-for-byte duplicate detection via SHA-256 of the trimmed text - fast, exact, no
# false positives (unlike the TF-IDF near-duplicate check below).
def check_exact_duplicates(records):
    seen = {}
    dups = []
    for r in records:
        h = hashlib.sha256(r["text"].strip().encode("utf-8")).hexdigest()
        if h in seen:
            dups.append((seen[h], r["id"]))
        else:
            seen[h] = r["id"]
    return dups


# Pairwise TF-IDF cosine similarity across every essay (O(n^2) pairs, fine at n=300) -
# catches essays that are substantially similar without being identical, e.g. human/
# hybrid pairs that legitimately share source text, or accidental near-copies.
def check_near_duplicates(records):
    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix)
    near_dups = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = sims[i, j]
            if score >= NEAR_DUP_THRESHOLD:
                near_dups.append((ids[i], ids[j], round(float(score), 3)))
    return near_dups


# Essays outside [WORD_COUNT_MIN, WORD_COUNT_MAX] on the raw `text` field - expected to
# fire heavily on `ai` essays specifically (see Fix 1 in DATASET.md), reported here
# rather than filtered, since the length-controlled field handles it non-destructively.
def check_word_count_outliers(records):
    return [(r["id"], r["label"], r["word_count"]) for r in records
            if r["word_count"] < WORD_COUNT_MIN or r["word_count"] > WORD_COUNT_MAX]


def check_encoding_and_whitespace(records):
    """Auto-fix trivial whitespace; report (don't fix) mojibake/control-char artifacts."""
    flagged = []
    fixed_count = 0
    for r in records:
        original = r["text"]
        cleaned = re.sub(r"[ \t]+", " ", original)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        if cleaned != original:
            r["text"] = cleaned
            r["word_count"] = len(cleaned.split())
            fixed_count += 1
        if MOJIBAKE_RE.search(cleaned) or CONTROL_CHAR_RE.search(cleaned):
            flagged.append(r["id"])
    return flagged, fixed_count


def check_mislabel_heuristic(records):
    """Flag surface-level oddities for manual review only (no auto-relabel)."""
    flagged = []
    for r in records:
        sents = r.get("sentence_labels")  # only present for hybrids
        text = r["text"]
        if r["label"] != "hybrid" and re.match(r"^\s*Title\s*:", text, re.IGNORECASE):
            flagged.append((r["id"], r["label"], "starts with 'Title:' header, atypical for this class"))
        if r["label"] == "human" and r["sentence_count"] and r["word_count"] / r["sentence_count"] > 35:
            flagged.append((r["id"], r["label"], "unusually long avg sentence length for a human essay"))
    return flagged


# Step 5: builds the markdown balance/diversity section of quality_report.md - class
# counts, word-count distributions (raw and length-controlled), AI-source diversity,
# hybrid sentence-diff coverage. Report-generation only, no checks/fixes of its own.
def balance_and_diversity_report(records):
    by_label = {"human": [], "ai": [], "hybrid": []}
    for r in records:
        by_label[r["label"]].append(r)

    lines = []
    lines.append("## Balance & Diversity (Step 5)\n")
    lines.append("| class | count |")
    lines.append("|---|---|")
    for label in ("human", "ai", "hybrid"):
        lines.append(f"| {label} | {len(by_label[label])} |")

    wc = {label: [r["word_count"] for r in recs] for label, recs in by_label.items()}
    lines.append("\n### Word count by class (raw `text` field)\n")
    lines.append("| class | min | max | mean | median |")
    lines.append("|---|---|---|---|---|")
    for label in ("human", "ai", "hybrid"):
        vals = wc[label]
        lines.append(
            f"| {label} | {min(vals)} | {max(vals)} | {statistics.mean(vals):.0f} | {statistics.median(vals):.0f} |"
        )

    lines.append(
        "\n**Note:** the raw `text` field still shows this length gap by design - it's left untouched for "
        "audit purposes. The gap is addressed non-destructively via the `text_length_controlled` field "
        "(1000-word cap, see `length_normalize.py`), which should be used for Phase 2 modeling instead "
        "of raw `text` if length parity matters.\n"
    )

    if "word_count_length_controlled" in records[0] if records else False:
        wc_lc = {label: [r["word_count_length_controlled"] for r in recs] for label, recs in by_label.items()}
        lines.append("### Word count by class (`text_length_controlled` field)\n")
        lines.append("| class | min | max | mean | median |")
        lines.append("|---|---|---|---|---|")
        for label in ("human", "ai", "hybrid"):
            vals = wc_lc[label]
            lines.append(
                f"| {label} | {min(vals)} | {max(vals)} | {statistics.mean(vals):.0f} | {statistics.median(vals):.0f} |"
            )
        lines.append("")

    lines.append("### AI source diversity\n")
    lines.append(
        "All 100 `ai` essays are recorded as `ai-generated (model/settings undocumented)` — "
        "single, unspecified model, no prompt/temperature record. No model diversity in this class; "
        "documented as a known limitation.\n"
    )

    diffable = sum(1 for r in by_label["hybrid"] if r.get("sentence_labels"))
    degenerate = sum(
        1 for r in by_label["hybrid"]
        if r.get("sentence_labels") and
        sum(1 for l in r["sentence_labels"] if l == "ai_edited") / max(len(r["sentence_labels"]), 1) >= 0.98
    )
    lines.append("### Hybrid sentence-diff coverage\n")
    lines.append(f"- {diffable}/{len(by_label['hybrid'])} hybrids have sentence-level diff labels against their human original.")
    lines.append(
        f"- {degenerate}/{diffable} of those show >=98% of sentences flagged `ai_edited`, meaning the Ollama "
        "hybrid-generation step rewrote nearly every sentence rather than preserving ~60% verbatim as the "
        "generation prompt intended. This is a real property of how class AH was produced, not a bug in the "
        "diff — documented as a known limitation.\n"
    )

    return "\n".join(lines)


# Entry point: runs every check_*() function above, auto-fixes only trivial whitespace
# issues (via check_encoding_and_whitespace), and writes dataset/quality_report.md with
# everything else reported for manual review rather than silently patched.
def main():
    records = load_records()

    print("=" * 60)
    print("STEP 4: DATA QUALITY PASS")
    print("=" * 60)

    empty_ids = check_empty(records)
    print(f"Empty/near-empty essays: {len(empty_ids)}")

    exact_dups = check_exact_duplicates(records)
    print(f"Exact duplicates: {len(exact_dups)}")

    print("Computing near-duplicate similarity (TF-IDF cosine)...")
    near_dups = check_near_duplicates(records)
    print(f"Near-duplicates (>= {NEAR_DUP_THRESHOLD}): {len(near_dups)}")

    outliers = check_word_count_outliers(records)
    print(f"Word-count outliers (<{WORD_COUNT_MIN} or >{WORD_COUNT_MAX} words): {len(outliers)}")

    encoding_flags, fixed_count = check_encoding_and_whitespace(records)
    print(f"Whitespace auto-fixed: {fixed_count} essays; encoding artifacts flagged: {len(encoding_flags)}")

    mislabel_flags = check_mislabel_heuristic(records)
    print(f"Mislabel-heuristic flags (manual review only): {len(mislabel_flags)}")

    save_records(records)

    print("\n" + "=" * 60)
    print("STEP 5: BALANCE & DIVERSITY")
    print("=" * 60)
    balance_section = balance_and_diversity_report(records)
    print(balance_section)

    with QUALITY_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Data Quality Report (Steps 4-5)\n\n")

        f.write("## Empty/near-empty essays\n\n")
        f.write("None found.\n\n" if not empty_ids else "\n".join(f"- {i}" for i in empty_ids) + "\n\n")

        f.write("## Exact duplicates\n\n")
        f.write("None found.\n\n" if not exact_dups else "\n".join(f"- {a} == {b}" for a, b in exact_dups) + "\n\n")

        f.write(f"## Near-duplicates (cosine >= {NEAR_DUP_THRESHOLD})\n\n")
        if near_dups:
            f.write("| essay A | essay B | similarity |\n|---|---|---|\n")
            for a, b, s in near_dups:
                f.write(f"| {a} | {b} | {s} |\n")
        else:
            f.write("None found.\n")
        f.write("\n")

        f.write(f"## Word-count outliers (outside {WORD_COUNT_MIN}-{WORD_COUNT_MAX} words, flagged not deleted)\n\n")
        if outliers:
            f.write("| id | label | word_count |\n|---|---|---|\n")
            for eid, label, wc_val in outliers:
                f.write(f"| {eid} | {label} | {wc_val} |\n")
        else:
            f.write("None found.\n")
        f.write("\n")

        f.write("## Encoding artifacts (not auto-fixed, needs manual review)\n\n")
        f.write("None found.\n\n" if not encoding_flags else "\n".join(f"- {i}" for i in encoding_flags) + "\n\n")
        f.write(f"(Trivial whitespace normalized automatically in {fixed_count} essays.)\n\n")

        f.write("## Mislabel heuristic flags (manual review only, nothing auto-relabeled)\n\n")
        if mislabel_flags:
            f.write("| id | label | reason |\n|---|---|---|\n")
            for eid, label, reason in mislabel_flags:
                f.write(f"| {eid} | {label} | {reason} |\n")
        else:
            f.write("None found.\n")
        f.write("\n")

        f.write(balance_section)

    print(f"\nWrote {QUALITY_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
