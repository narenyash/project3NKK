# AI Essay Detector — Backend API

A FastAPI service wrapping the two trained models from Phase 2 (`PHASE2_PROCESS.md`) and
evaluated in Phase 4 (`EVALUATION.md`). This is a wiring/integration layer — it does not
retrain, retune, or reimplement any feature or model logic; it imports directly from
`scripts/features.py` and `scripts/text_utils.py`.

## Starting the service

```
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Startup loads GPT-2, spaCy, both trained models, and `dataset/corpus_baseline.json`
**once**, at process start (takes a few seconds). Every request after that only runs
inference — nothing reloads per-request. Check `GET /health` once the process is up to
confirm everything loaded before sending real traffic.

## Why two models

The essay-level model (7 features) is strong at spotting **wholesale AI-written essays**
but was found in Phase 2/4 to be nearly blind (20% accuracy) to **AI-edited sentences
hidden inside an otherwise-human essay** — the realistic "a person wrote this, then had
it polished" case. A second, sentence-level-only model (2 features:
`perplexity_z`, `logprob_variance_z`) was built specifically for that case and recovers
76% recall on it (held-out test data) — at the cost of a much higher false-positive rate
on ordinary text (56%, rising to 77% on non-native English writing). **Neither model
should be read as a standalone verdict.** They answer different questions and both carry
real, quantified error rates — every score in the API response ships with its own
confidence note stating the actual number, not a generic disclaimer.

## `GET /health`

```json
{ "status": "ok", "models_loaded": true }
```

## `POST /analyze`

Request:
```json
{ "text": "essay text here..." }
```

Validation:
- Empty text → 400.
- Under 50 words → 400. Both models were trained/evaluated on essay-length text; the
  essay-level features (burstiness, repetition) need multiple sentences to mean anything
  at all, and a single sentence or two isn't representative of what was validated.
- Over 5,000 words → 400. One GPT-2 forward pass runs per sentence — very long inputs
  scale response time linearly and this cap keeps it bounded.

Response shape:
```json
{
  "essay_level": {
    "ai_likelihood_score": 0.84,
    "confidence_note": "..."
  },
  "sentences": [
    {
      "index": 0,
      "text": "...",
      "paragraph_index": 0,
      "sentence_level_score": 0.52,
      "confidence": "low",
      "confidence_note": "...",
      "sentence_model_features": [
        {"feature": "perplexity_z", "value": -0.05, "raw_value": 14.32, "contribution": 0.01, "direction": "toward_human"},
        {"feature": "logprob_variance_z", "value": -0.45, "raw_value": 0.87, "contribution": 0.03, "direction": "toward_human"}
      ],
      "top_features": [
        {"feature": "sentence_length_variance_z", "value": -1.60, "raw_value": 6.21, "contribution": 2.96, "direction": "toward_ai"},
        {"feature": "perplexity_z", "value": 0.31, "raw_value": 18.05, "contribution": 0.18, "direction": "toward_ai"},
        "... all 7 essay-model features, sorted by contribution descending"
      ],
      "essay_model_local_score": 0.96,
      "essay_model_local_score_note": "..."
    }
  ],
  "paragraphs": [
    {
      "index": 0, "sentence_count": 4, "ai_likelihood_score": 0.91,
      "local_outlier": {
        "mean_perplexity_deviation": 1.8,
        "sentence_length_variance_deviation": -0.4,
        "syntactic_depth_variance_deviation": 2.1,
        "notable": true
      }
    }
  ],
  "paragraph_outlier_note": "...",
  "fairness_note": "...",
  "sentence_count": 33
}
```

Field notes:
- `essay_level.ai_likelihood_score` — mean of the essay-level model's per-sentence
  probability across the essay, thresholded nowhere (raw 0-1 score, frontend's call how
  to present a cutoff). This is the number with the most validation behind it.
- `sentences[].sentence_level_score` — the sentence-level model's own output. Always
  ships with `confidence: "low"` and a note stating its real FPR (56% / 77% ESL) — this
  is intentional and should not be dropped or softened by the frontend; it's a direct
  response to the Phase 4 fairness finding, not boilerplate.
- `sentences[].top_features` — all 7 essay-model features for that sentence, sorted by
  `contribution` (`|coefficient × z-score|`) descending, used for the "why" explanation.
  Each entry also carries `direction`: `"toward_ai"` if `coefficient × z-score > 0`,
  `"toward_human"` if `< 0`, `"neutral"` if ~0 — this tells the frontend which way *this
  essay's* value on that feature actually pushed the score, without re-deriving the sign
  itself. `raw_value` is the feature's untransformed value (before z-scoring against the
  baseline) — e.g. actual perplexity, actual sentence-length variance — for readers who
  want the literal number behind the z-score, not just its normalized form.
- `sentences[].sentence_model_features` — the sentence-level model's own 2 raw features
  in full (not just "top"), so a skeptical reader can verify the flag's own thin
  evidence directly rather than trusting the confidence note's word for it. Same
  `direction` and `raw_value` fields as above.
- `sentences[].essay_model_local_score` — the essay-level model's per-sentence
  probability *before* averaging into the essay-level score. Useful for a heatmap-style
  "which sentences pulled the verdict" view, but explicitly **not independently
  evaluated** as a sentence-level detector — its own note says so. Don't present this as
  equivalent in reliability to `ai_likelihood_score` or `sentence_level_score`.
- `fairness_note` — always present, states the ESL false-positive-rate finding plainly.
  Should be visible in the UI, not buried.
- `sentences[].paragraph_index` — which paragraph (blank-line-separated block in the
  submitted text) this sentence belongs to. Computed by matching the already-segmented
  sentence list against paragraph boundaries in the original text — sentence segmentation
  itself is unchanged, this only labels sentences after the fact, so it can't diverge from
  the validated segmentation behavior used elsewhere.
- `paragraphs` — one entry per paragraph: `sentence_count` and `ai_likelihood_score` (mean
  of that paragraph's sentences' `essay_model_local_score`, same averaging pattern as
  `essay_level.ai_likelihood_score`, just scoped to one paragraph). Exists for the
  "sentences AND passages" requirement — a paragraph can read as AI-associated in
  aggregate even when no single sentence in it is individually extreme. Same reliability
  caveat as `essay_model_local_score`: not independently evaluated on its own, it's a
  rollup of an already-validated per-sentence number, not a separately-trained/tested
  detector.
- `paragraphs[].local_outlier` — **exploratory heuristic, not run through either trained
  model, not compared against the global baseline**: how far this paragraph's own mean
  perplexity / sentence-length variance / syntactic-depth variance deviate from the
  *other paragraphs in this same essay*. `notable: true` when any `|deviation| > 1.5`.
  Only present when the essay has 2+ paragraphs (a deviation from "the rest of the
  essay" is undefined with one paragraph); individual deviation fields can still be
  `null` for a paragraph with fewer than 2 sentences (variance over one sentence is
  degenerate, excluded rather than shown as a false zero). **Deliberately framed as a
  magnitude signal ("statistically different"), never a direction toward AI or human** —
  `PHASE2_PROCESS.md` Step 7.5 found a related idea (sentence-vs-neighborhood deviation,
  used as a *trained* feature) pointed the wrong way: human sentences deviated from their
  neighbors *more* than AI-edited ones. That finding doesn't directly apply here (this is
  shown raw, never classified), but it's exactly why the frontend must not imply "flagged
  = AI."
- `paragraph_outlier_note` — present whenever `paragraphs[].local_outlier` is populated
  (2+ paragraphs), states the above caveat in reader-facing language. Should always be
  shown alongside any `local_outlier` badge in the UI, not dropped.

## Known behavior (not a bug)

The essay-level model has a real, documented ~20% false-positive rate on genuine human
essays, driven mostly by one dominant feature (`sentence_length_variance_z`) that reads
consistently-short/consistent-length human writing as AI-like. This fires often in
practice — see `EVALUATION.md`'s addendum for a live, reproducible example (a hand-written
essay about a grandmother's kitchen scored 0.84). Frontend copy should never present
`ai_likelihood_score` as a verdict ("this essay IS AI-written") — prefer language like
"shows patterns statistically associated with AI-written text," consistent with the
actual, imperfect reliability documented above.

## What's not implemented here

Authentication, persistence, rate limiting, and batching are all out of scope for this
step — single stateless endpoint, in-memory models, intended for local/internal use while
the frontend is built against it.
