"""
Phase 2, Step 1: build a sentence-level table from the Phase 1 essay dataset.

Unit of analysis for Phase 2 is one row per sentence, not per essay. Sentences are
segmented to match `text_length_controlled` (not raw `text`) per the Phase 2 spec,
since raw `text` still carries the pre-Phase-1-fix length disparity between classes.

Implementation note: rather than re-splitting the already-truncated
`text_length_controlled` string, we re-derive the same word-capped sentence prefix
directly from a fresh split of the original `text`. `length_normalize.py` rejoins kept
sentences with plain spaces, which discards paragraph/newline structure; re-splitting
that flattened string can shift spaCy's sentence boundaries for essays containing things
like numbered lists (confirmed on 10/98 hybrids during a dry run - e.g. "forward: 1. How
can..." got merged into one sentence instead of two once the original line break was
gone). Applying the identical truncation algorithm to the original full-text sentence
list avoids that round-trip entirely and guarantees the result stays a true prefix of
`sentence_labels` (which was diffed against the same full-text split in Phase 1).

For human/ai essays, every sentence inherits the essay-level label. For hybrid essays,
each sentence gets its own label from the Phase 1 `sentence_labels` diff array (human /
ai_edited) - these are not relabeled here, only sliced to the truncated length.
"""
import json
import re
from pathlib import Path

from length_normalize import WORD_CAP
from text_utils import split_sentences, word_count

# Junk-fragment filter (Phase 2, Step 4a). Numbered-list items (e.g. "1. How can...")
# sometimes get split by spaCy into a bare numeral fragment ("1.") plus the rest of the
# item as a separate "sentence." Citation/reference markers (e.g. "RE1", "RE5") get
# similarly mis-segmented into standalone fragments - discovered via a Step 3 sanity
# check, where two such fragments in one essay (h_066) inflated GPT-2 perplexity to
# 129,308 and 54,662, distorting that essay's perplexity_variance to 358M (vs a normal
# range of ~1,700-130,000 across the rest of the 30-essay sample) and skewing the whole
# human-group mean through it. Neither pattern is real sentence content.
#
# Generalized to one pattern: an optional short (0-4 char) letter prefix, followed by
# 1-4 digits, optional trailing period, and NOTHING else (anchored, no spaces allowed -
# a fragment must be the entire "sentence," not just contain a number). This is
# deliberately narrow: it requires at least one digit, so it can never match a real
# all-letters short sentence like "Cost matters." or "Explore." (verified against every
# <=2-word sentence found in Phase 1's Step 1 pass - none contain digits).
#
# Matches:  "2."    "13"    "RE1"   "RE5"   "R12."   "AB1234"
# Does NOT match: "Cost matters."   "Explore."   "How can..."   "APUSH"   "COVID-19"
#   (APUSH/COVID-19 aren't pure letter-prefix+digit - "COVID-19" has a hyphen, "APUSH"
#   has no digit at all)
BARE_FRAGMENT_RE = re.compile(r"^[A-Za-z]{0,4}\d{1,4}\.?$")

# Second pattern found at full-scale (Phase 2, Step 4d follow-up): venue/reference tokens
# like "NeurIPS'21" - same species as the citation markers above (a bare label picked up
# as its own "sentence"), just with an apostrophe before the digits instead of none. This
# is a SEPARATE, narrower pattern from BARE_FRAGMENT_RE (which has no apostrophe in it)
# rather than a broadening of it, so real contractions are never at risk: this pattern
# requires digits immediately after the apostrophe, which no English contraction has
# ("don't", "There's", "O'Brien" - none are letters+apostrophe+DIGITS).
#
# Matches: "NeurIPS'21"   "NeurIPS'21."   "ICML'20"
# Does NOT match: "don't"   "There's"   "O'Brien"   "can't stop"
CITATION_APOSTROPHE_RE = re.compile(r"^[A-Za-z]+['’]\d{1,4}\.?$")

ROOT = Path(__file__).resolve().parent.parent
ESSAYS_PATH = ROOT / "dataset" / "essays.jsonl"
SPLITS_PATH = ROOT / "dataset" / "splits.json"
OUT_PATH = ROOT / "dataset" / "sentences.jsonl"


def truncate_sentence_list(sentences):
    """Mirrors length_normalize.truncate_to_cap's algorithm, but returns the list of
    kept sentences instead of a rejoined string, so callers never need to re-split."""
    kept = []
    total = 0
    for s in sentences:
        w = word_count(s)
        if total + w > WORD_CAP and kept:
            break
        kept.append(s)
        total += w
    return kept


def load_essays():
    """All 300 records from dataset/essays.jsonl (Phase 1's output), excluded ones included -
    build_rows_for_essay() below is what actually drops them."""
    return [json.loads(line) for line in ESSAYS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split_lookup():
    """essay_id -> "train"/"test", flattened from splits.json's two id lists into one dict
    for O(1) lookup per essay below."""
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    lookup = {}
    for eid in splits["train"]:
        lookup[eid] = "train"
    for eid in splits["test"]:
        lookup[eid] = "test"
    return lookup


# One essay in, its sentence-level rows out - the core per-essay transformation this
# whole script exists to do. Handles all three essay labels: human/ai sentences all
# inherit their essay's label; hybrid sentences use Phase 1's own sentence_labels diff.
def build_rows_for_essay(essay, split_lookup, warnings):
    eid = essay["id"]
    if essay.get("excluded"):
        return []
    if eid not in split_lookup:
        warnings.append(f"{eid}: not present in splits.json (excluded elsewhere?) - skipping")
        return []

    split = split_lookup[eid]
    topic = essay.get("topic")
    essay_label = essay["label"]

    full_sentences = split_sentences(essay["text"])
    lc_sentences = truncate_sentence_list(full_sentences)

    # Cross-check against length_normalize.py's own word-count result for this essay,
    # to confirm our re-derivation matches what was actually persisted in essays.jsonl.
    lc_word_count = sum(word_count(s) for s in lc_sentences)
    if lc_sentences == full_sentences and essay["word_count"] != essay["word_count_length_controlled"]:
        warnings.append(f"{eid}: essay wasn't truncated here (word cap not reached) but "
                         "word_count_length_controlled differs from word_count in essays.jsonl")

    if essay_label in ("human", "ai"):
        sentence_labels = [essay_label] * len(lc_sentences)
    else:  # hybrid
        full_labels = essay.get("sentence_labels") or []
        if len(full_labels) != len(full_sentences):
            warnings.append(
                f"{eid}: sentence_labels length ({len(full_labels)}) != full-text sentence "
                f"count ({len(full_sentences)}) - label alignment may be off"
            )
        sentence_labels = full_labels[: len(lc_sentences)]

    rows = []
    for idx, (sent_text, sent_label) in enumerate(zip(lc_sentences, sentence_labels)):
        rows.append({
            "essay_id": eid,
            "sentence_idx": idx,
            "sentence_text": sent_text,
            "sentence_label": sent_label,
            "topic": topic,
            "split": split,
            "essay_label": essay_label,
        })
    return rows


# Entry point: builds every essay's rows, filters junk fragments, re-indexes, writes
# dataset/sentences.jsonl, and prints a summary (counts by label/split, any warnings).
def main():
    essays = load_essays()
    split_lookup = load_split_lookup()

    print("=" * 60)
    print("PHASE 2 STEP 1: BUILD SENTENCE-LEVEL TABLE")
    print("=" * 60)

    warnings = []
    all_rows = []
    for essay in essays:
        all_rows.extend(build_rows_for_essay(essay, split_lookup, warnings))

    pre_filter_count = len(all_rows)

    # True for a bare numeral/citation fragment (BARE_FRAGMENT_RE or CITATION_APOSTROPHE_RE
    # above) that should be dropped rather than treated as real sentence content.
    def is_junk(text):
        t = text.strip()
        return bool(BARE_FRAGMENT_RE.match(t) or CITATION_APOSTROPHE_RE.match(t))

    numeral_rows = [r for r in all_rows if is_junk(r["sentence_text"])]
    all_rows = [r for r in all_rows if not is_junk(r["sentence_text"])]
    # Re-index sentence_idx per essay after dropping rows, so indices stay contiguous.
    counters = {}
    for r in all_rows:
        counters.setdefault(r["essay_id"], 0)
        r["sentence_idx"] = counters[r["essay_id"]]
        counters[r["essay_id"]] += 1

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    excluded_ids = {e["id"] for e in essays if e.get("excluded")}
    present_essay_ids = {r["essay_id"] for r in all_rows}
    assert not (excluded_ids & present_essay_ids), "excluded essays leaked into sentences.jsonl"

    print(f"Filtered out {len(numeral_rows)} bare fragments (numerals + citation/reference-style):")
    for r in numeral_rows:
        print(f"  {r['essay_id']}: {r['sentence_text']!r}")
    if not numeral_rows:
        print("  (none found)")
    print(f"Wrote {len(all_rows)} sentence rows to {OUT_PATH.relative_to(ROOT)} "
          f"(from {pre_filter_count} before filtering)")
    print(f"Confirmed 0 rows from excluded essays ({sorted(excluded_ids)} correctly absent)")

    from collections import Counter
    by_label = Counter(r["sentence_label"] for r in all_rows)
    by_split = Counter(r["split"] for r in all_rows)
    by_split_label = Counter((r["split"], r["sentence_label"]) for r in all_rows)

    print("\nRow counts by sentence_label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")

    print("\nRow counts by split:")
    for split, count in sorted(by_split.items()):
        print(f"  {split}: {count}")

    print("\nRow counts by (split, sentence_label):")
    for (split, label), count in sorted(by_split_label.items()):
        print(f"  {split} / {label}: {count}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nNo warnings - all hybrid label alignments were clean prefixes.")


if __name__ == "__main__":
    main()
