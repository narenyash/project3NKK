# EVALUATION.md — Held-Out Test Set Results

This is the first and only time `dataset/splits.json`'s test-split essays (60 essays,
never read, inspected, or used for anything until this document) were used. Nothing in
either model was retrained or retuned based on these results. Numbers here are final.

Full build history: `PHASE1_PROCESS.md` (dataset), `PHASE2_PROCESS.md` (feature
engineering and modeling). Dataset spec and limitations: `dataset/DATASET.md`.

---

## What's being evaluated

Two separate, purpose-built models (Phase 2, Steps 7 and 7.6):

- **Essay-level model** (`logistic_regression_model.joblib`) — 7 z-scored features
  (`perplexity`, `logprob_variance`, `sentence_length_variance`,
  `syntactic_depth_variance`, `perplexity_variance`, `ngram_repeat_rate`,
  `transition_repetition_rate`). Answers "is this whole essay likely AI-written."
- **Sentence-level model** (`sentence_level_model.joblib`) — only `perplexity_z` and
  `logprob_variance_z`, the two features that actually vary sentence-to-sentence.
  Answers "does this specific sentence look AI-touched," built specifically because the
  essay-level model was blind (20% accuracy) to AI-edited sentences hiding inside
  otherwise-human hybrid essays.

Both were z-scored using the existing `dataset/corpus_baseline.json` (fit once on
train-split human data, never touched again) — not recomputed for this evaluation.

---

## Step 1 — Test-split preparation

Test-split essays had never had features extracted before this step (though their raw
sentence text was already segmented with all fixes applied, since the fix logic lives in
the shared `text_utils.split_sentences()` used by every essay regardless of split).

Fix-impact counts, test split specifically:

| fix | test essays affected |
|---|---|
| Tab-N document header strip | 30/60 |
| Bare numeral/citation fragment filter | 7 fragments (`h_050` ×5, `hy_087` ×1, `hy_092` ×1) |
| Apostrophe mis-split fix | 1 essay (`hy_087`) |

3,498 sentence rows extracted, z-scored against the existing baseline, clipped to
`[-5, 5]`. Output: `dataset/features_test_zscored.jsonl`.

---

## Step 2 — Essay-level model on the test set

### Sentence-level evaluation (directly comparable to Step 7 validation)

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Step 7 validation | 83.98% | 91.68% | 83.28% | 87.28% |
| **Test set** | **82.73%** | **91.74%** | **82.38%** | **86.81%** |

Confusion matrix `[[TN=907, FP=179], [FN=425, TP=1987]]`.

Per-label accuracy:

| label | n | validation | **test** | delta |
|---|---|---|---|---|
| human | 1,086 | 85.35% | **83.52%** | -1.83 pts |
| ai | 1,885 | 99.87% | **99.95%** | +0.08 pts |
| ai_edited | 527 | 20.00% | **19.54%** | -0.46 pts |

**The model generalizes cleanly** — every number lands within 2 points of validation,
confirming Step 7's results weren't overfit to that particular slice.

### Essay-level aggregation (new for this step)

Step 7 never aggregated to one verdict per essay — it only trained/evaluated at sentence
granularity. **Method chosen: mean predicted probability across all sentences in the
essay, thresholded at 0.5.** Reasoning: 5 of the 7 features are essay-wide constants (the
same value on every sentence in an essay), so per-sentence probabilities within one essay
are already highly correlated — averaging is a natural, low-variance summary rather than
an arbitrary majority vote.

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Essay-level aggregation | 65.00% | 85.19% | 57.50% | 68.66% |

Confusion matrix `[[TN=16, FP=4], [FN=17, TP=23]]` (n=60 essays: 20 human, 20 ai, 20 hybrid).

Per-essay-label accuracy:

| label | n | accuracy |
|---|---|---|
| human | 20 | 80.00% |
| ai | 20 | **100.00%** |
| hybrid | 20 | **15.00%** |

**This low hybrid number is expected, not a bug.** A hybrid essay is majority-human by
sentence count (most sentences are verbatim-retained human text), so averaging correctly
pulls the essay-level verdict toward "human" — the model is accurately reflecting that
most of a hybrid essay's content genuinely is human-written. This essay-level model was
never intended to catch hybrids at the whole-essay level; that's the sentence-level
model's job specifically. Reported here in full rather than hidden, because it's an
important, honest characterization of what this aggregation method can and can't do.

---

## Step 3 — Sentence-level model on the test set

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Step 7.6 validation | 59.08% | 69.53% | 67.57% | 68.54% |
| **Test set** | **58.43%** | **71.95%** | **65.09%** | **68.35%** |

Confusion matrix `[[TN=474, FP=612], [FN=842, TP=1570]]`.

Per-label accuracy:

| label | n | validation | **test** | delta |
|---|---|---|---|---|
| human | 1,086 | 42.62% | **43.65%** | +1.03 pts |
| ai | 1,885 | 66.60% | **62.02%** | -4.58 pts |
| **ai_edited** | 527 | **71.28%** | **76.09%** | **+4.81 pts** |

### The central question: did `ai_edited` recall hold up?

**Yes — it held up, and slightly improved**, from 71.28% (validation) to 76.09% (genuinely
unseen test essays). This is not an artifact of the specific validation slice; it's a
real, reproducible effect of decoupling the sentence-level features from the essay-wide
ones. This is the single most important result of Phase 4: the two-tier fix from Step 7.6
holds on data neither model has ever seen.

---

## Step 4 — Three confidently-wrong examples

### 1. Essay-level model: a human essay misread as AI because it's naturally low-burstiness

`hy_012` ("The Translating College Essay Example," a short montage-style essay) —
4 of the top 5 most-confidently-wrong human predictions all come from this single essay:

> `hy_012#3`: *"As I look back on my life, I realized that this was my first act of
> translation."* — proba(AI) = **0.7992**, true = human

Features: `sentence_length_variance=35.62`, `syntactic_depth_variance=2.74` — both low.
This essay is short and consistently structured (a montage of short parallel anecdotes),
which naturally produces low sentence-to-sentence variance — exactly the statistical
signature the model was trained to associate with AI text. **This is the essay-wide
feature dominance problem, seen from the failure side**: a legitimately human essay that
happens to write in a tight, consistent register gets every one of its sentences flagged,
because the dominant coefficient can't tell "genuinely low-variance human writing" apart
from "AI-generated low-variance writing." It's a structural blind spot, not a one-off.

### 2. Essay-level model: an AI-edited passage hidden by a bibliography section

`hy_052` — an academic-style hybrid essay containing a citation list. 5 sentences,
including genuinely AI-edited ones, were scored at essentially proba(AI) ≈ 0.001–0.002
(i.e., near-certain human) despite the true label being `ai_edited`:

> `hy_052#21`: *"MIT Technology Review."* — perplexity=69.96, true=ai_edited, proba(AI)=0.0016
> `hy_052#16`: *"Machine bias: There's software used across the country to predict future
> criminals."* — perplexity=851.61, true=ai_edited, proba(AI)=0.0012

Individual sentence perplexity is high (as expected for citation fragments and dense
academic prose), but the essay-wide `sentence_length_variance=163.23` and
`syntactic_depth_variance=10.72` are both *very* high — because a citation list creates
huge artificial length variation (a 2-word citation next to a 40-word academic sentence).
**High burstiness is supposed to signal "human," and the model correctly learned that —
but here the burstiness is a structural artifact of an academic essay's bibliography
section, not genuine human stylistic variation**, and it's confident enough to overrule
even very high per-sentence perplexity.

### 3. Sentence-level model: AI-generated aphorisms with unexpectedly *high* perplexity

The model's core assumption is "AI text = lower perplexity." Several genuinely AI-written
sentences violate this directly:

> `ai_010#37`: *"Explore."* — perplexity=**1881.38**, true=ai, proba(AI)=0.2799
> `ai_002#21`: *"Cost matters."* — perplexity=**1881.67**, true=ai, proba(AI)=0.2976
> `ai_016#52`: *"Success can hide assumptions."* — perplexity=**3240.66**, true=ai, proba(AI)=0.3009

These are short, punchy, aphoristic one-liners — a deliberate rhetorical device the source
model uses for emphasis. GPT-2 finds them highly surprising precisely *because* they're
unusual, terse phrasing choices, not because they're statistically "AI-like" in the sense
the detector assumes. **This shows perplexity captures "how unusual is this phrasing,"
not "was this written by AI," and those two things point in opposite directions whenever
an AI model deliberately writes in a stylistically unusual register.**

---

## Step 5 — ESL fairness check

Sample: 300 of 3,911 ELLIPSE corpus essays (English Language Learner writing, all
genuinely human), seeded random sample (seed=42) — disclosed explicitly rather than
silently substituted for the full corpus, given runtime constraints of processing all
3,911 (5,680 sentences extracted from the 300-essay sample).

**Result: false-positive rate is elevated on ESL writing across all three evaluation
modes, confirming the exact concern the project brief flagged.**

| | Main test set (genuine human) FPR | ELLIPSE (ESL) FPR | Delta |
|---|---|---|---|
| Essay model, sentence-level | 16.48% | **32.45%** | +16.0 pts (essentially doubled) |
| Essay model, essay-level | 20.00% | **26.33%** | +6.3 pts |
| Sentence-level model | 56.35% | **77.39%** | +21.0 pts |

**This is a real, meaningful fairness gap, not noise.** The essay-level model's
sentence-level false-positive rate roughly *doubles* on ESL writing (16.5% → 32.5%), and
the sentence-level model — already prone to false positives on native-English human text
(56.4%) — flags over three-quarters of ESL sentences as AI-touched (77.4%). Both models
show the pattern; neither should be treated as fairness-neutral.

**Plausible mechanism**: several of the features that drive these models — perplexity,
log-prob variance, sentence-length variance, syntactic-depth variance — are all, at root,
measures of how *predictable* or *conventional* a sentence is to GPT-2, which was trained
overwhelmingly on fluent native-English text. Non-native phrasing (unconventional word
order, simplified sentence structure, non-idiomatic word choices) is exactly the kind of
thing that reads as "unusual" or "low-burstiness" to a model like this — the same
statistical signature the system was built to associate with AI text. This isn't a
guess specific to this project; it's the documented failure mode these kinds of
perplexity/burstiness-based detectors are known for, and this test confirms this system
is not exempt from it.

**Implication for deployment**: neither model should be used to make high-stakes,
unreviewed decisions about ESL writers without this bias being surfaced to the reviewer,
and ideally without further work specifically aimed at closing this gap (e.g. an
ESL-specific baseline, or excluding/down-weighting the features most responsible for it)
before any real-world use. This is reported here, not fixed, per this phase's explicit
scope.

---

## Step 6 — Honest summary

**What the system is good at**: catching wholesale AI-generated essays. The essay-level
model is essentially perfect on pure `ai` essays (99.95% sentence-level, 100% essay-level
on this test set) and solid on genuine human essays (83–85%).

**A real fairness problem exists and is not fixed**: both models show a meaningfully
elevated false-positive rate on ESL (non-native English) writing — roughly double for the
essay-level model (16.5%→32.5%) and severe for the sentence-level model (56.4%→77.4%).
This is exactly the failure mode the project brief warned these kinds of detectors have a
habit of exhibiting, and this system is not exempt from it. It should not be deployed for
unreviewed, high-stakes decisions about ESL writers without this being surfaced and,
ideally, addressed first.

**What it's weaker at**: sentence-level detection of AI-edited content inside an
otherwise-human essay — the case the project brief calls "the realistic case." The
two-tier fix (Step 7.6) took this from a 20% blind spot to 76% recall on unseen data,
which is real, confirmed, non-trivial progress — but it comes at a real cost (the
sentence-level model's overall accuracy is only 58%, well below the essay-level model's
83%, because with only 2 modest features it can't cleanly separate `human` from `ai` the
way the full feature set can). It should be read as a flag/signal, not a confident verdict,
and used *alongside* the essay-level model, not instead of it.

**Explicitly out of scope / unresolved**: DetectGPT-style perturbation scoring and a
larger reference language model were identified in `PHASE2_PROCESS.md` as the next real
step if more sentence-level lift is needed — neither was attempted in Phase 2 or here.
Changing the hybrid-generation dataset was tested directly (two independent diagnostic
experiments, `PHASE2_PROCESS.md` §11) and ruled out as a fix.

**Dataset and methodology limitations carried forward** (full detail in `DATASET.md` and
`PHASE2_PROCESS.md`):
- AI essay source model/settings are undocumented (single unspecified model, no prompt
  record).
- Topic tagging is unreliable at the distribution level (94% of essays landed in one
  topic under 3-model consensus) — stratification by topic should not be trusted.
- A residual title-fusion effect (essay titles glued onto the real first sentence) was
  found but deliberately left unfixed after discovering that naive line-based stripping
  would have deleted entire essay bodies for some AI-rewritten hybrids — a smaller, known,
  accepted source of noise in sentence-0 features specifically.
- The dataset is small (296 usable essays, 60 held out for this test) — all numbers above
  should be read with that sample size in mind, especially the essay-level aggregation
  breakdown (n=20 per class).
- The ESL fairness check used a 300-essay sample of the 3,911-essay ELLIPSE corpus (5,680
  sentences), not the full corpus, for runtime tractability — a large, seeded, disclosed
  sample, but not exhaustive. The direction and scale of the finding (elevated FPR across
  all three evaluation modes, +6 to +21 points) is unlikely to be a sampling fluke given
  its consistency across all three, but the exact percentages could shift somewhat on the
  full corpus.

---

## Addendum — a 4th confidently-wrong example, found live during backend integration

Not one of the original 3 examples above (those came from the held-out test set during
Phase 4); this one surfaced while testing the backend pipeline (`backend/pipeline.py`) on
a fresh, hand-written essay that was never part of the dataset at all — which makes it
arguably *stronger* evidence of the pattern than an in-dataset example, since there's no
possibility it's an artifact specific to how this dataset was built.

**Essay-level model: a hand-written human essay about a grandmother's kitchen, scored
0.8421 AI-likelihood** (essay-level model, live inference, no test-split involvement):

> *"My grandmother's kitchen smelled like burnt garlic and cardamom, and I hated it until
> I was about fourteen..."* (9 sentences, ~180 words, genuinely human-written for this
> test, never seen by either model in training or evaluation)

Every sentence in the essay was flagged with `sentence_length_variance_z = -1.6044` as
its top contributing feature — the same dominant coefficient (-1.84, by far the largest
in the essay-level model) responsible for wrong-example #1 above (`hy_012`). This essay
happens to use fairly consistent sentence lengths throughout (a stylistic choice, not
unusual for reflective personal narrative), which reads to the model as the same
low-burstiness signature associated with AI text.

**Why this is worth keeping as a reference case**: it's a live, reproducible
demonstration that the essay-level model's ~20% false-positive rate on human text (Step 2)
isn't a rare edge case confined to one dataset essay — it will surface again, readily, on
ordinary fresh human writing with a consistent style. Anyone presenting this tool's output
to an end user should expect to see this exact failure mode in practice, not treat it as
a hypothetical documented in a report.
