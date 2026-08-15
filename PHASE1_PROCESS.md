# Phase 1 — Full Process Documentation

This document is a step-by-step record of everything done in Phase 1 of the AI-detector
project: from the raw essay data (a mix of self-collected submissions and a public essay
collection — see below) through to a clean, documented, analysis-ready dataset. It covers
what was done, in what order, with which tool/script, and why — including the problems
found along the way and how they were fixed.

For the dataset's final schema, counts, and known limitations, see `dataset/DATASET.md`.
This document is about the *process*, not the dataset spec.

For a short, plain-language retelling of this same story (no stats/ML background
assumed), see the "How this tool actually works" section at the top of the live app
(`frontend/src/components/HowItWorks.jsx`) — this document is the full technical record
it's summarized from.

---

## 0. Starting point — what existed before Phase 1

Three raw folders, 100 files each, no shared schema, no documentation:

| Folder | Contents | Naming |
|---|---|---|
| `Class A/` | 100 AI-generated essays | `1.txt` … `100.txt` |
| `class H/` | 100 human-written essays — **mixed provenance**: `Tab_1`–`Tab_28` self-collected student submissions; `Tab_29`–`Tab_100` pulled from the public essay collection at [openessays.org](https://www.openessays.org/) via `split_open_source_essays.py` (see below) | `Tab_1.txt` … `Tab_100.txt` |
| `class AH/` | 100 "hybrid" essays (human + AI polish) | `Tab_1_hybrid.txt` … `Tab_100_hybrid.txt` |

**Correction note**: earlier drafts of this document (and of `dataset/DATASET.md`)
described `class H` as entirely self-collected. That was inaccurate — confirmed by file
timestamps (`Tab_1`–`Tab_28` all carry the original collection pass's timestamp;
`Tab_29` onward all carry the later timestamp from the `split_open_source_essays.py` run,
which also overwrote what had briefly been a self-collected `Tab_29.txt`). Both this
document and `DATASET.md` have been corrected to state the real 28/72 split.

Also present from earlier work (not part of this pipeline, kept as-is; later moved into
`legacy_raw_data_setup/` during a documentation/organization pass — see that folder's
`README.md`): `create_hybrid_essays.py`, `create_hybrid_open_source_essays.py`,
`move_hybrid_essays.py`, `split_tabs.py`, `setup_ollama_model.py`, `test_ollama.py` —
these were the original scripts used to build the raw `class AH` hybrids and split source
text files. They were inspected to understand how the raw data was produced, but were not
modified. `split_open_source_essays.py` (also present from earlier work) is the one that
pulled `OPEN_SOURCE_ESSAYS.txt` — the openessays.org dump — into `class H/Tab_29.txt`
onward; called out separately here since it directly affects how `human` essays should be
attributed, not just a housekeeping script.

**Tooling available:** Python 3.12, a local Ollama installation with three models already
pulled (`mistral:latest`, `llama3:latest`, `gemma4:latest`), no internet API keys. No
`pandas`, `spaCy`, or `scikit-learn` installed yet.

---

## 1. First pass — building the pipeline (Steps 1–9)

The goal of this pass: turn the three raw folders into one normalized, documented dataset.
Nine sub-steps, each with its own script under `scripts/`:

### Step 0 — Environment setup
Installed `pandas`, `scikit-learn` (for TF-IDF near-duplicate detection), and `spacy` +
the `en_core_web_sm` model (for sentence segmentation). The spaCy model download
succeeded (internet was available), so sentence splitting used spaCy throughout rather
than falling back to the regex splitter also built into `scripts/text_utils.py`.

### Step 1–2 — `scripts/build_dataset.py`
- **Inventoried** all three folders: confirmed 100/100/100 files, checked for empty
  files, and confirmed every `class H` tab number has a matching `class AH` hybrid
  (needed for sentence-diffing hybrids against their source).
- **Normalized** every essay into one JSON-Lines record with a shared schema: `id`
  (`h_001`, `ai_001`, `hy_001`, etc.), `text`, `label` (`human`/`ai`/`hybrid`), `topic`
  (filled in later), `genre`, `source`, `word_count`, `sentence_count` (filled in later).
- **Sentence-diffed every hybrid against its human original** using `difflib.SequenceMatcher`
  over spaCy sentence spans, producing a `sentence_labels` array per hybrid essay
  (`human` = verbatim sentence, `ai_edited` = new/altered by the polish step). This is
  what let us later measure exactly how much of each hybrid was genuinely a blend versus
  a full rewrite.
- Output: `dataset/essays.jsonl` (300 records) and `dataset/conversion_report.md` (issues
  found during normalization).

### Step 3 — `scripts/topic_tagger.py` (first version)
Prompted a single local Ollama model to classify each essay into one of 8 Common-App-style
topic categories, with a self-reported confidence level. Output: topic written back into
`essays.jsonl`, plus `dataset/topic_coverage.csv` (topic × class counts) and
`dataset/topic_low_confidence.md` (essays flagged below high confidence).

### Step 4–5 — `scripts/quality_checks.py`
- Checked for empty/near-empty essays, exact duplicates (hash), near-duplicates (TF-IDF
  cosine similarity via scikit-learn, threshold 0.85), word-count outliers (outside
  250–1000 words), encoding artifacts, and a surface-level mislabel heuristic.
- Auto-fixed only trivial whitespace issues; everything else was reported, not silently
  changed.
- Produced a balance/diversity summary: class counts, word-count distribution, AI-source
  diversity, hybrid sentence-diff coverage.
- Output: `dataset/quality_report.md`.

### Step 6 — `scripts/sentence_segment.py`
Filled in `sentence_count` for all 300 essays using the same spaCy splitter used for the
Step 2 diff, so the two stayed consistent.

### Step 7 — `scripts/make_split.py`
Built an essay-level 80/20 train/test split, stratified by `(label, topic)`, seeded for
reproducibility, and locked it into `dataset/splits.json`.

### Step 8 — `scripts/fetch_ellipse.py`
Downloaded the public [ELLIPSE corpus](https://github.com/scrosseye/ELLIPSE-Corpus)
(English Language Learner writing samples, ~3,911 essays, no auth required) and normalized
it into a standalone `dataset/ellipse_fairness.jsonl` — kept completely separate from the
main dataset, reserved for a later fairness/false-positive-rate check (never used for
training).

### Step 9 — `dataset/DATASET.md`
Wrote the first version of the dataset documentation: final counts, sourcing methodology
per class, and a **Known Limitations** section — written honestly rather than silently
patched, per the original instruction to flag rather than paper over problems.

### What the first pass surfaced (Known Limitations v1)
1. **Length disparity** — `ai` essays averaged ~3,200 words vs. ~1,000 for `human`/`hybrid`.
2. **Hybrid generation missed its target** — 51/100 hybrids came back from the original
   whole-essay Ollama rewrite with 0% verbatim sentence retention (not the ~60% intended),
   including two (`hy_021`, `hy_026`) that were fully fabricated because their human
   "source" (`h_021`, `h_026`) was a 2-word stub, not a real essay.
3. **Topic tagging was unreliable** — a single model claimed high confidence on all 300
   essays while dumping 81% into one category (`intellectual_curiosity`), including 74%
   of the human essays — clearly an overused default, not a trustworthy signal.

---

## 2. Second pass — fixing the three flagged problems

The user asked to actually fix these three issues rather than leave them as documented
limitations. Decisions made before implementing (via direct Q&A):
- Length fix: **non-destructive** (add a normalized field, don't touch raw text).
- Stub essays (`h_021`/`h_026`/`hy_021`/`hy_026`): **exclude** from the modeling-ready set.
- Hybrids: **regenerate all 98 valid ones** with a new, more controlled method (not just
  the 51 broken ones), so the whole class is built consistently.
- Topics: **3-model consensus** (mistral, llama3, gemma4 — all already available locally).

### Fix 1 — `scripts/regenerate_hybrids.py` (new script)
Replaced the old "rewrite the whole essay" approach with a controlled blend:
- Split each essay into paragraphs (`\n\n`); if an essay didn't have real paragraph
  breaks (30/98 essays didn't), fall back to sentence-level units instead.
- Deterministically select ~40% of those units (seeded random, reproducible) and send
  **only those** to Ollama (`mistral:latest`) for enhancement — the rest are copied
  through byte-for-byte.
- Reassemble in original order and overwrite `class AH/Tab_N_hybrid.txt`.
- **Before overwriting**, the entire original `class AH/` folder was copied to
  `class AH_original_backup/` (100 files) so nothing from the first pass was lost.
- `Tab_21` and `Tab_26` were skipped (no valid human source to blend from).
- This ran as ~736 individual Ollama calls (one per selected unit across 98 essays) and
  took roughly 20–30 minutes.

### Fix 2 — `scripts/length_normalize.py` (new script)
Added a non-destructive length-controlled variant:
- Chose a **1000-word cap** (close to the human median of ~968 words).
- For every essay, truncated `text` at the last full sentence at or before the cap;
  essays already under the cap were left as-is (no padding).
- Stored as new fields `text_length_controlled` / `word_count_length_controlled`,
  alongside the untouched original `text`/`word_count`.
- Result: `ai` mean dropped from 3,203 → 995 words; `human` 998 → 826; `hybrid`
  (which had grown to 1,597 after regeneration, since AI-enhanced sentences tend to
  expand) → 948. All three classes now sit in a comparable range.

### Fix 3 — `scripts/topic_tagger.py` (rewritten)
Switched from one model to **3-model consensus**:
- Each of mistral, llama3, and gemma4 independently tags every essay.
- If 2-or-3 models agree, that topic is kept with `topic_confidence: "high"`.
- If all three disagree, the essay is flagged `topic_confidence: "low"` and all three
  guesses are stored in a new `topic_candidates` field for manual review.
- **Performance fix mid-run**: the first attempt looped essay-outer, model-inner (call
  mistral, then llama3, then gemma4, for essay 1; repeat for essay 2; …), which forced
  Ollama to swap each multi-GB model in and out of memory ~900 times on a machine with
  very little free RAM (observed ~700MB free out of 15.6GB total). This was killed and
  rewritten to loop **model-outer**: load mistral once and tag all 300 essays, then load
  llama3 and tag all 300, then gemma4 — cutting the number of model loads from ~900 to 3
  and making the run dramatically faster. Also switched to unbuffered stdout (`python -u`)
  so progress could be watched live in the terminal instead of appearing only at the end.
- Full run: 300 essays × 3 models = 900 Ollama calls, ~1–1.5 hours total (gemma4, the
  largest model at ~9.6GB, ran noticeably slower than the other two).

### Rebuilding downstream artifacts
After the three fixes, the rest of the pipeline was re-run in the correct order so
everything stayed consistent:
1. `regenerate_hybrids.py` — rebuilt `class AH/`
2. `build_dataset.py` — rebuilt `essays.jsonl` from the new hybrid text; added
   `excluded` / `exclusion_reason` fields for `h_021`/`h_026`/`hy_021`/`hy_026`
3. `length_normalize.py` — added the length-controlled fields
4. `sentence_segment.py` — refilled `sentence_count`
5. `topic_tagger.py` — 3-model consensus topic tags
6. `quality_checks.py` — rerun on the new data (also patched to report both raw and
   length-controlled word-count tables, so the report wouldn't read as a stale, unresolved
   risk after Fix 1)
7. `make_split.py` — updated to filter out `excluded: true` records before splitting, and
   the old `splits.json` was deleted and regenerated (an intentional re-lock, since the
   underlying data had materially changed)

### What improved vs. what's still a limitation (see `dataset/DATASET.md` for full detail)
- **Length**: fixed, non-destructively. ✅
- **Hybrid blending**: fixed — every valid hybrid now has genuine partial retention
  (17.7%–81.9% verbatim, mean 41.5%) instead of the old all-or-nothing pattern. ✅
  (Residual note: mean retention came out somewhat more AI-heavy than the 60/40 target,
  because AI-enhanced units tend to expand into more sentences than the original.)
- **Topic tagging confidence**: fixed — the confidence signal is now honest (4/300 essays
  correctly flagged as genuinely ambiguous, vs. 0/300 falsely-confident before). ✅
- **Topic tagging distribution**: *not* fully fixed — even with 3-model consensus, 94% of
  essays landed on `intellectual_curiosity` under majority vote (up from 81% with a single
  model). Documented honestly as a residual, unresolved limitation rather than claimed as
  solved. ⚠️
- **Stub essays**: excluded from the modeling-ready set (4 records: `h_021`, `h_026`,
  `hy_021`, `hy_026`), kept in `essays.jsonl` for audit. Usable dataset: 296 essays. ✅

---

## 3. Tools and libraries used

| Tool | Purpose |
|---|---|
| Python 3.12 | All scripting |
| `pandas` | Installed for potential tabular work (not heavily used directly — JSONL was the primary format) |
| `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`) | Near-duplicate detection in `quality_checks.py` |
| `spaCy` (`en_core_web_sm`) | Sentence segmentation, used for both `sentence_count` and the hybrid sentence-diff |
| `difflib.SequenceMatcher` (standard library) | Sentence-level diffing between human originals and hybrids |
| `ollama` (Python client) | All LLM calls — hybrid essay generation (mistral) and topic tagging (mistral + llama3 + gemma4) |
| `urllib` (standard library) | Downloading the ELLIPSE corpus CSV from GitHub |
| `csv`, `json` (standard library) | Reading/writing all output formats |

---

## 4. Final file map

```
Class A/                       raw AI essays (100, untouched)
class H/                       raw human essays (100, untouched)
class AH/                      hybrid essays (100, REGENERATED in the second pass)
class AH_original_backup/      the pre-fix hybrid essays (100, audit copy)

scripts/
  text_utils.py                 shared sentence-splitting / word-count helpers
  build_dataset.py               Steps 1-2: inventory, normalize, hybrid sentence diff, exclusions
  regenerate_hybrids.py          controlled paragraph/sentence blend for class AH (fix)
  length_normalize.py            adds text_length_controlled field (fix)
  sentence_segment.py            Step 6: sentence counts
  topic_tagger.py                Step 3: 3-model consensus topic tagging (fix)
  quality_checks.py              Steps 4-5: dedup/outlier/balance checks
  make_split.py                  Step 7: locked train/test split, excludes stubs
  fetch_ellipse.py               Step 8: ELLIPSE fairness slice download

dataset/
  essays.jsonl                   the 300-record normalized dataset (296 usable, 4 excluded)
  conversion_report.md           Step 1-2 issues
  quality_report.md              Step 4-5 quality/balance findings
  topic_coverage.csv             topic x class counts
  topic_low_confidence.md        4 essays with genuine 3-way topic disagreement
  splits.json                    locked 80/20 train/test split (296 records)
  ellipse_fairness.jsonl         standalone ELL fairness slice (3,911 essays)
  DATASET.md                     dataset spec + limitations (the "what", this doc is the "how")

PHASE1_PROCESS.md                this document
```

*(This map reflects Phase 1's own deliverables at the time it was written — it doesn't
track everything added since, e.g. `backend/`, `frontend/`, `Class A_v2/`, or the
`legacy_raw_data_setup/` folder the pre-Phase-1 one-off scripts were later archived into.
See the root `README.md` for the current, full project layout.)*

---

## 5. What Phase 1 delivered

A 296-essay (usable), 300-essay (raw, audit-inclusive) dataset of human/AI/hybrid college
admissions essays, with:
- A shared, documented schema (`essays.jsonl`)
- Sentence-level provenance for every hybrid essay
- A non-destructive length-controlled variant to prevent trivial length-based classification
- Topic tags with an honest confidence signal (even though the underlying distribution is
  still skewed and flagged as such)
- A locked, stratified, leakage-free train/test split
- A separate fairness-check corpus (ELLIPSE) reserved for later bias evaluation
- Full documentation of both the dataset itself (`DATASET.md`) and the process used to
  build it (this file)

Ready for Phase 2 (feature extraction / model training) once that phase's scope is defined.
