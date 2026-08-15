# Phase 2 — Full Process Documentation

This is a step-by-step record of Phase 2: building a feature-extraction pipeline and an
interpretable AI-detection model on top of the Phase 1 dataset. It covers what was tried,
what broke, what was fixed, what was tested, and the final honest state of the project —
including a real, evidence-backed limitation that Phase 2 did not solve.

For Phase 1 (dataset finalization), see `PHASE1_PROCESS.md` and `dataset/DATASET.md`.
This document assumes that dataset (296 usable essays, 300 raw, locked train/test split)
as its starting point.

For a short, plain-language retelling of this same story (no stats/ML background
assumed), see the "How this tool actually works" section at the top of the live app
(`frontend/src/components/HowItWorks.jsx`) — this document is the full technical record
it's summarized from.

---

## 0. Ground rules carried over from Phase 1

- Local, open-source model only for measurement (GPT-2 base via `transformers`/`torch`) —
  never a chat/instruction model producing a verdict directly.
- Ollama models (mistral/llama3/gemma4) stay confined to *generation* tasks (already used
  in Phase 1 for hybrid essays and topic tagging) — never used in the detection pipeline.
- Only the **train split** is touched anywhere in Phase 2. The held-out test split is
  reserved for a later, separate evaluation phase and was never read from.
- Every step reports honestly — counter-intuitive or negative results are surfaced, not
  smoothed over or silently patched.

---

## 1. Step 0 — GPT-2 setup and sanity check

Installed `transformers`, `torch`, `numpy`. Hit a real environment problem immediately:
the first `torch` install (version 2.13, CPU wheel) had a packaging bug — its own code
imported a `torchgen` submodule that isn't actually published as an installable package,
so the import chain broke (`ModuleNotFoundError: No module named 'torchgen.model'`). A
force-reinstall didn't help because a *previous* broken install had no `RECORD` file, so
pip couldn't cleanly uninstall it either. Fixed by manually deleting the broken
`torch`/`torchgen`/`functorch` directories from site-packages and pinning to a known-good
stable release, `torch==2.5.1`.

Wrote `scripts/gpt2_sanity_check.py`: loads GPT-2 base, runs one forward pass on a
hardcoded sentence, extracts per-token log-probabilities from the raw logits (via
`log_softmax`), and computes perplexity from them. Also hit a Windows-console encoding
issue (`cp1252` can't render GPT-2's byte-level BPE markers like `Ġ`) — fixed by forcing
`sys.stdout.reconfigure(encoding="utf-8")` in every script that prints tokenizer output.
Confirmed working: 15-token test sentence, perplexity 37.09, sane per-token log-probs
(predictable words like "the" scored near 0, unusual ones like "cat" scored around -9).

## 2. Step 1 — Sentence-level table

Wrote `scripts/build_sentences.py`. Unit of analysis for all of Phase 2 is one row per
**sentence**, not per essay, since hybrid essays need per-sentence `human`/`ai_edited`
labels (already computed in Phase 1's `sentence_labels` diff array).

- Sentences segmented from `text_length_controlled` (the length-capped Phase 1 field),
  re-derived directly from a fresh split of the original `text` rather than round-tripping
  through the already-truncated string — `length_normalize.py` rejoins kept sentences with
  plain spaces, which was found to shift spaCy's sentence boundaries for essays containing
  numbered lists.
- Found and filtered a class of junk fragments: numbered-list items (e.g. `"1. How
  can..."`) sometimes get split by spaCy into a bare numeral fragment (`"1."`) plus the
  rest as a separate "sentence." A regex (`BARE_FRAGMENT_RE`) catches these — cross-checked
  against all 495 short (≤3 word) sentences in the corpus with zero false positives before
  applying at scale. 24 fragments removed initially.
- Output: `dataset/sentences.jsonl` — 17,326 rows (later refined further, see Step 4).

## 3. Step 2 — Four feature functions

Wrote `scripts/features.py`. All four are pure functions; GPT-2 is loaded once and cached.

**Functions 1 & 2 — perplexity & log-prob variance**: share a single GPT-2 forward pass
per sentence (`sentence_token_logprobs()`) — context (up to 3 preceding sentences from the
same essay) and the target sentence are tokenized *separately* and concatenated as
token-ID sequences, rather than concatenating text and masking labels, so the
context/sentence boundary is known exactly regardless of BPE merging. `sentence_perplexity()`
and `token_logprob_variance()` both derive from this one call.

**Function 3 — essay-level burstiness**: three sub-metrics computed once per essay —
sentence-length variance, syntactic-parse-tree-depth variance (via spaCy dependency
parses), and perplexity variance across the essay's sentences.

**Function 4 — n-gram diversity/repetition**: went through four design iterations before
landing on something that showed real signal:
- v1: bigram/trigram type-token ratio (TTR) over a fixed 4-sentence window — pinned near a
  1.0 ceiling (0.978–1.0), too little text for repeats to occur.
- v2: same idea, widened to a ~120-word window — barely moved (0.975–1.0).
- v3: v2 + stripped stopwords before building n-grams — made it *worse*, trigram TTR went
  completely flat at 1.0 everywhere, since fewer tokens per window made 3-word sequences
  even more likely to be unique by chance.
- v4 (final): abandoned local-window TTR entirely. Counts exact bigram/trigram repeats
  across the **whole essay** and reports a repeat *rate* (not a ratio) — real, interpretable
  spread confirmed on spot-tests (0.010–0.042 across sample essays, e.g. a research-topic
  essay repeating "language models" 7 times scored meaningfully higher than others).

Spot-tests surfaced an important interpretive note: perplexity tracks token-level *rarity*
(unusual words/proper nouns), not topical distinctiveness — a sentence about an ordinary
personal topic phrased in plain words can score lower perplexity than a sentence with an
unusual name in it, regardless of how "distinctive" the underlying idea is.

## 4. Step 3 — Stratified sanity check (go/no-go gate)

30-essay stratified sample (10 human/10 ai/10 hybrid, train split only, seed=42). Ran all
four functions and checked 7 directional predictions (do AI-influenced sentences show
lower perplexity, lower log-prob variance, lower burstiness, lower n-gram repetition than
human — the hypothesized signature of AI text).

Result: 5/7 nominally matched, but digging into `perplexity_variance`'s "match" revealed
it was a single-essay artifact (see Step 4) — the real, artifact-free picture was 4/7
solid matches and 3/7 genuine non-matches (`perplexity_variance`, `ngram_repeat_rate`,
`transition_repetition_rate`).

## 5. Step 4 — Segmentation bug hunt and full-scale extraction

Two real, previously-invisible data-quality bugs were found and fixed here, both via the
same method: an implausibly large essay-level statistic led to inspecting the actual text.

**Bug 1 — citation/reference fragments.** Essay `h_066` contained bare citation markers
(`"RE1"`, `"RE5"`) mis-segmented as standalone sentences. GPT-2 scored them at perplexity
129,308 and 54,662, inflating that one essay's `perplexity_variance` to 358 million — which
alone was skewing the entire human-group mean (35.9 million). Fixed by extending the
Step 1 junk-fragment filter to a generalized pattern: `^[A-Za-z]{0,4}\d{1,4}\.?$` (an
optional short letter prefix + digits + optional period, and *nothing* else) — verified
against 17 hand-picked test cases and confirmed it can never match a real all-letters short
sentence like `"Cost matters."` (no digit present).

**Bug 2 — apostrophe mis-split.** spaCy's sentence segmenter mis-split a contraction into
its own "sentence" when it used a curly apostrophe (U+2019) instead of straight — e.g.
`"There's no denying its wow factor."` broke into `"There's"` + `"no denying its wow
factor."` Fixed in `text_utils.split_sentences()`: segmentation boundaries are computed on
a straight-apostrophe-normalized copy of the text, but the returned sentences are sliced
from the *original* string by character offset, so original quote characters are preserved
in the output — only the segmentation decision changes. Also incidentally fixed a related
case (`"NeurIPS'21"`, a bare venue/year token) without needing a second regex.

Full-scale extraction over the entire train split (13,811–13,818 sentences depending on
which pass, 236 essays) confirmed the artifact-free result: **4/7 checks match** (perplexity,
log-prob variance, sentence-length variance, syntactic-depth variance); **3/7 genuinely
don't** (`perplexity_variance` — confirmed inverted, ai *higher* than human, not lower;
`ngram_repeat_rate`; `transition_repetition_rate`).

## 6. Step 5 — Corpus baseline and z-score normalization

Single **global** human baseline (not per-topic) — an explicit, stated consequence of
Phase 1's finding that topic tagging is unreliable at the distribution level (94% of
essays landed in one topic under 3-model consensus; most topic buckets were too small to
produce a stable baseline).

Caught and fixed a real bug in this step's own first draft: the essay-level baseline
(burstiness/repetition features) was accidentally computed over "any essay containing a
human-labeled sentence," which pulled in all 78 hybrid essays alongside the 78 genuine
human essays (156 "human essays" instead of 78) — because hybrids contain human-labeled
sentences too (their verbatim-retained portions). Fixed by scoping essay-level features to
`essay_label == "human"` specifically, while sentence-level features (perplexity, log-prob
variance) correctly stayed scoped by `sentence_label` alone (a verbatim human sentence
inside a hybrid essay is still legitimately human-written at the sentence level).

Output: `dataset/corpus_baseline.json` (mean/std per feature) and
`dataset/features_train_zscored.jsonl`. Zero directional flips vs. the raw-value results,
as expected for a linear transform.

## 7. Step 6 — Weighted-sum baseline scorer

A simple, transparent, non-tuned reference point before the real model. All 7 z-score
columns clipped to `[-5, 5]` (459 values clipped total, concentrated in
`perplexity_variance_z` — 315 of 459 — driven by one known outlier essay, `ai_019`, whose
extreme value was independently verified as legitimate content, not a bug).

Weights: equal magnitude (-1.0) on the 4 established-direction features only
(`perplexity_z`, `logprob_variance_z`, `sentence_length_variance_z`,
`syntactic_depth_variance_z`); the 3 non-matching features excluded from this simple
baseline (their fate deferred to the real model's trained coefficients).

Result: **76.6% accuracy, 85.4% precision, 78.6% recall** — a real, working signal from
just perplexity + burstiness, no repetition features at all.

## 8. Step 7 — Logistic regression, and the central discovery

Binary target: `human=0`, `ai`/`ai_edited`=1. Essay-stratified 80/20 train-sub/validation
split *within* the train split (48 of 236 essays held out, seed=42 — the true test split
untouched throughout). `class_weight='balanced'`, converged cleanly.

**Coefficients** (sorted by magnitude): `sentence_length_variance_z` (-1.84, by far the
largest) → `syntactic_depth_variance_z` (-0.37) → `perplexity_z` (-0.25) →
`transition_repetition_rate_z` (-0.19) → `logprob_variance_z` (-0.14) →
`perplexity_variance_z` (+0.08) → `ngram_repeat_rate_z` (+0.02). All three flagged
features (5/6/7) recommended **drop** — near-zero relative to the dominant feature.

**Validation metrics looked good in aggregate**: 83.9% accuracy, 91.8% precision, 83.0%
recall — better than the Step 6 baseline across the board.

**The central discovery, from the per-label breakdown the spec insisted on**:

| label | accuracy |
|---|---|
| human | 85.5% |
| ai | 99.7% |
| **ai_edited** | **19.2%** |

The model is effectively blind to AI-edited sentences hiding inside otherwise-human hybrid
essays — worse than random for that specific subgroup. Root cause, visible directly in the
coefficients: of the 7 features, only `perplexity_z` and `logprob_variance_z` vary
sentence-to-sentence. The other 5 — including the single dominant coefficient — are
**essay-wide constants** joined onto every sentence in an essay. Since hybrid essays'
overall burstiness already looks human-like, that essay-wide signal swamps whatever weak
sentence-level signal exists for any individual AI-edited sentence.

Saved: `dataset/logistic_regression_model.joblib` (reloadable) and
`dataset/logistic_regression_model.json` (plain coefficients, usable for inference without
sklearn).

## 9. Step 7.5 — Attempting a fix: sentence-local features

Purpose: add genuinely **sentence-local** features (computed relative to a sentence's own
nearby neighborhood, not the whole essay or the global baseline) to give the model
something that can actually vary within a hybrid essay.

Two new functions added to `features.py`:
- `local_perplexity_deviation()` — how far a sentence's perplexity deviates from the mean
  of the *other* sentences in the same essay (excluding itself).
- `local_burstiness_disruption()` — word-count and syntactic-depth deviation from a ±2
  rolling window of neighboring sentences (clipped at essay boundaries, no padding).

**Spot-test surfaced a major new data-quality finding** before the feature idea was even
tested at scale (see Section 10). After that got fixed, the honest result — confirmed via
both a corrected spot-test and a full 13,816-sentence extraction — was:

| feature | human-within-hybrid \|mean\| | ai_edited \|mean\| |
|---|---|---|
| local_perplexity_deviation | 1.398 | 0.558 |
| local_length_deviation | 1.635 | 1.580 |
| local_depth_deviation | 1.468 | 1.427 |

All three point the **wrong way** — human sentences deviate from their local neighborhood
*more* than AI-edited ones, not less. A full retrain (same 48-essay validation split as
Step 7) confirmed this hurts, not helps: overall accuracy dropped (83.98% → 83.12%), and
**`ai_edited` accuracy got worse, not better** (20.00% → 15.62%). `local_perplexity_deviation_z`
picked up a real coefficient (-0.40) reflecting the actual (backwards) pattern, and in
doing so cannibalized `perplexity_z`'s own signal through multicollinearity (that
coefficient flipped from -0.24 to a near-zero, wrong-signed +0.03).

Per explicit instruction, no further window-size or feature-definition tuning was
attempted without checking in — this was reported as a genuine negative result.

## 10. The "Tab N" header artifact (mid-Step-7.5 detour)

While spot-testing the new local features, an unrelated, previously-unknown, and much
larger-scoped bug surfaced: **149 of 296 essays** had their literal source-file header
(e.g. `"Tab 50"`, left over from the original `Tab_N.txt` file-splitting in Phase 1) baked
directly into sentence index 0 as if it were essay content. In one case (`hy_050`), this
inflated that sentence's perplexity to 80.5x its local neighborhood — not because of
anything related to authorship, but because GPT-2 finds a bare document label wildly
improbable as sentence text.

**Characterization**: 150 of 151 affected first-lines were the *exact* literal pattern
`"Tab N"` and nothing else (extremely uniform) — never isolated as its own sentence,
always fused onto the front of the real content because there's no punctuation between
them (only a newline, which spaCy doesn't treat as a sentence boundary). **Severely
skewed by label**: `ai` 0/100 affected (0%), `human` 95/98 (97%!), `hybrid` 54/98 (55%) —
a near-universal, one-directional bias hitting the human class almost exclusively.

**Fix**: a new pre-segmentation regex in `text_utils.split_sentences()`
(`_TAB_HEADER_RE = r"^Tab \d+[ \t]*\n"`), stripping only an exact whole-line match — verified
it correctly leaves alone a case where the AI-rewrite step had legitimately transformed
`"Tab 18"` into a real title (`"Tab 18: Unleashing Creativity through Code"`, which has
trailing text after the number and so doesn't match).

**Blast-radius quantification** (before committing to a full re-run): targeted
recomputation on the 118–119 affected train-split essays showed individual sentence-0
perplexity dropped substantially (mean 63 points, up to 5.5x for one essay), but the
*global* human baseline barely moved (-1.4%) — diluted across ~4,500 total human
sentences. The human-ai perplexity *gap* shrank by a more meaningful -14.5% relative,
though. Given the severe label skew (a named trigger for a full re-run) and an
acknowledged, unquantified risk to the dominant `sentence_length_variance` feature, a full
re-run of Steps 4/5/6/7 was recommended and approved.

**A second, smaller residual issue was found and deliberately left unfixed**: even after
stripping "Tab N", many essays' title line (e.g. `"THE 'What Rocks Tell Us' COLLEGE ESSAY
EXAMPLE"`) is still fused onto the real first sentence, producing a smaller but real
perplexity elevation (144–437, vs. a normal ~50–100 range). Attempting to strip this too
was judged too risky to do quickly and safely: it was discovered that for some hybrid
essays, the AI-rewrite step in Phase 1 had collapsed the original newline structure
entirely (confirmed on `hy_002`, where a naive "strip line 2" approach would have deleted
the *entire essay body*, not just the title). This was documented as a known, accepted
limitation rather than rushed.

**Full re-run result**: coefficients and validation metrics moved only marginally
(accuracy 83.87%→83.98%, essentially unchanged). Crucially, **`ai_edited` accuracy barely
moved** (19.23%→20.00%) — confirming the Tab-N bug, while real and worth fixing for data
integrity, was never the cause of the blind spot. The dominant `sentence_length_variance_z`
coefficient barely changed at all (-1.8371→-1.8441).

## 11. Diagnostic experiments: is the hybrid dataset itself the problem?

Prompted by the question "would changing the hybrid dataset fix this," two independent
experiments were run, neither touching the production dataset:

**Experiment 1 — regenerate without the voice-matching instruction.** The production
hybrid-generation prompt explicitly asks the AI to "keep the same core content, meaning,
and voice" when polishing a paragraph — a plausible reason AI-edited sentences blend in so
well. 8 essays were regenerated with a stripped prompt (no voice/length-matching
instruction) via `scripts/experiment_no_voice_match.py`, written to a separate
`experiment_hybrids/` folder. Result: aggregate separation barely changed (experimental
human=2.11/ai_edited=0.60 vs. production human=2.23/ai_edited=0.49) — still the wrong
direction, just marginally less extreme. One essay (`h_006`) did reverse dramatically, but
that was 1 of 8, not a general pattern.

**Experiment 2 — test the user's own manually-created hybrids.** Before Phase 2's
automated regeneration, `class AH_original_backup/` preserved the original hybrid files —
some of which (tabs 98, 99, 100) were manually created by the user via GPT rather than the
automated mistral pipeline. Tested these plus 2 randomly-selected tabs (33, 78) against
production. Found 3 of 5 backup essays were actually **100% rewritten** (no human
sentences retained at all — not real blends), so no comparison was possible for those. The
2 that did retain human content (`h_078`, `h_099`) showed the **same wrong-direction
pattern** as production.

**Combined verdict**: the wrong-direction pattern is robust across three independently-
generated hybrid sources (mistral with voice-matching, mistral without it, and manual
GPT). Changing *how* the hybrid essays were generated does not fix the separation problem.
This points to something more fundamental than a dataset-construction artifact: AI-authored
sentences, regardless of model or instruction, appear to be **locally more statistically
consistent with their neighbors** than human sentences are — close to the opposite of what
"local perplexity deviation" was built to detect. This is now a well-evidenced conclusion,
not a guess.

---

## 12. Where Phase 2 stands

**Working well**: essay-level AI-detection is strong. `ai` essays are caught with 99.7%+
accuracy; overall binary accuracy is ~84% with 92% precision. Four features
(`perplexity_z`, `logprob_variance_z`, `sentence_length_variance_z`,
`syntactic_depth_variance_z`) carry real, well-tested, artifact-free signal.

**Not working**: sentence-level detection of AI-edited content hiding inside an
otherwise-human essay — the scenario the project's own brief calls "the realistic case."
Two genuinely different feature designs were tried (Step 2's essay-wide burstiness, Step
7.5's sentence-local deviation) and both failed at this specific sub-task; the second one
made it measurably worse by interfering with an otherwise-working feature. Three
independently-sourced hybrid datasets all show the same negative result, ruling out
"bad training data" as the explanation.

**Two real, unrelated data-quality bugs were found and fixed** along the way (citation
fragments, apostrophe mis-splits, and the much larger-scoped Tab-N header issue) — all
improving data integrity, none of them the cause of the sentence-level blind spot.

**A solution for the sentence-level blind spot is proposed separately** — see the
project's solution-proposal note for the concrete recommendation and reasoning.

## 13. Tools and files added in Phase 2

| Tool | Purpose |
|---|---|
| `transformers`, `torch` (pinned 2.5.1) | GPT-2 loading and inference |
| `scikit-learn` `LogisticRegression` | The interpretable detector |
| `joblib` | Model persistence |
| spaCy (`en_core_web_sm`, already from Phase 1) | Sentence segmentation, dependency parsing |

```
scripts/
  gpt2_sanity_check.py         Step 0
  build_sentences.py           Step 1 (+ Step 4a junk-fragment filter extension)
  features.py                  Step 2 (4 functions) + Step 7.5 (2 local functions)
  step2_spot_test.py           Step 2 spot-tests
  step3_sanity_check.py        Step 3
  step4_full_extraction.py     Step 4c
  step4d_report.py             Step 4d
  step_tabfix_blast_radius.py  Tab-N fix blast-radius quantification
  step5_baseline_normalize.py  Step 5
  step6_baseline_scorer.py     Step 6
  step7_train_model.py         Step 7
  step7_5_spot_test.py         Step 7.5a
  step7_5b_full_local.py       Step 7.5b
  step7_5c_normalize.py        Step 7.5c
  step7_5d_retrain.py          Step 7.5d
  experiment_no_voice_match.py Diagnostic experiment 1
  analyze_experiment_hybrids.py  Diagnostic experiment 1 analysis
  test_backup_hybrids.py       Diagnostic experiment 2

dataset/
  sentences.jsonl                        Step 1 output (refined through Step 4)
  features_train_raw.jsonl               Step 4c full-scale raw features
  corpus_baseline.json                   Step 5 baseline (+ Step 7.5c local-feature extension)
  features_train_zscored.jsonl           Step 5 z-scored features
  features_train_clipped.jsonl           Step 6 clipped features
  baseline_scores_train.jsonl            Step 6 weighted-sum scores
  logistic_regression_model.joblib/.json Step 7 trained model
  features_train_with_local*.jsonl       Step 7.5 local-feature tables

experiment_hybrids/                      Diagnostic experiment 1 output (not production data)
```
