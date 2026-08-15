"""
Step 2: the full inference pipeline, wrapped as one function. Combines segmentation
(with all Phase 2 fixes), 7-feature extraction, z-scoring against the fixed baseline,
both model inferences, and a per-sentence "top contributing features" explanation.

No retraining, no threshold changes, no reimplementation of feature math - everything
here calls straight into scripts/features.py and scripts/text_utils.py.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import features as F
from text_utils import split_sentences
from build_sentences import BARE_FRAGMENT_RE, CITATION_APOSTROPHE_RE

from model_loader import get_state, ESSAY_MODEL_FEATURES, SENTENCE_MODEL_FEATURES, FPR_STATS

CLIP_LO, CLIP_HI = -5.0, 5.0

# Human-readable confidence notes, sourced directly from EVALUATION.md's numbers.
ESSAY_CONFIDENCE_NOTE = (
    "Based on whole-essay statistical patterns (sentence length/syntax variance, overall "
    "perplexity). Strong on fully AI-written essays (99.95% sentence-level accuracy, 100% "
    "essay-level on held-out test data) and solid on genuine human essays (~83-85%). "
    "False-positive rate on ordinary human writing is ~20% (essay-level), rising to ~26% "
    "on non-native English (ESL) writing - see the fairness note."
)
SENTENCE_CONFIDENCE_NOTE = (
    "This flag targets AI-edited sentences hidden inside otherwise-human essays specifically "
    "(76.09% recall on held-out test data for that case). Looking at only one sentence at a "
    "time is inherently harder than looking at the whole essay, so this signal often lands "
    "near the midpoint for ordinary writing - that's expected, not a malfunction. Its "
    "false-positive rate on ordinary human writing is 56.35%, rising to 77.39% on non-native "
    "English (ESL) writing, so treat it as one supporting data point alongside the essay-wide "
    "score above, not a verdict on its own."
)
ESSAY_LOCAL_SCORE_NOTE = (
    "This is the essay-level model's contribution from this specific sentence, before "
    "averaging into the essay-level score above. Useful for visualizing which sentences "
    "pulled the essay verdict up or down, but it has NOT been independently evaluated as "
    "a sentence-level detector on its own - only the averaged essay-level score and the "
    "separate sentence-level model have been validated against held-out test data."
)
FAIRNESS_NOTE = (
    "This tool has shown reduced accuracy for non-native English (ESL) writers: "
    "false-positive rates roughly double for the essay-level flag (16-20% -> 26-32%) and "
    "rise further for the sentence-level flag (56% -> 77%) on ESL writing in evaluation. "
    "Do not use this tool's output as the sole basis for a decision about a non-native "
    "English writer."
)

MIN_WORDS = 50
MAX_WORDS = 5000


_WHITESPACE_RE = re.compile(r"\s+")


def _clean_sentences(text: str) -> list[str]:
    """
    Segment + apply the same junk-fragment filter used throughout Phase 2. Also collapses
    internal whitespace (line-wraps within a single sentence span) to single spaces for
    display - this is purely cosmetic text hygiene applied AFTER segmentation, it doesn't
    change split_sentences()'s segmentation decisions or the apostrophe-fix character-
    offset logic in text_utils.py.
    """
    raw = split_sentences(text)
    cleaned = [_WHITESPACE_RE.sub(" ", s).strip() for s in raw]
    return [s for s in cleaned if not (BARE_FRAGMENT_RE.match(s) or CITATION_APOSTROPHE_RE.match(s))]


# (raw_value - baseline_mean) / baseline_std, clipped to [CLIP_LO, CLIP_HI] - the exact
# same z-scoring + clipping used to build corpus_baseline.json in Phase 2's
# step5_baseline_normalize.py, applied here at inference time against that same fixed
# baseline rather than recomputed from the current request's text.
def _zscore(raw_value, field, baseline):
    if raw_value is None:
        return None
    mean, std = baseline[field]["mean"], baseline[field]["std"]
    if std == 0:
        return None
    z = (raw_value - mean) / std
    return max(CLIP_LO, min(CLIP_HI, z))


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Paragraph text blocks, split on blank lines in the ORIGINAL submitted text (before
    sentence segmentation). Whitespace-collapsed the same way _clean_sentences() collapses
    sentence text, so the substring matching in _assign_paragraphs() lines up. This does
    NOT re-run sentence segmentation - it only produces paragraph boundaries used to label
    the sentence list _clean_sentences() already produced, so it can't diverge from the
    validated segmentation behavior relied on elsewhere (Phase 2/4, training).
    """
    raw = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [_WHITESPACE_RE.sub(" ", p).strip() for p in raw if p.strip()]


def _assign_paragraphs(sentences: list[str], paragraphs: list[str]) -> list[int]:
    """
    Greedy positional match: walk sentences and paragraphs together in document order,
    advancing the paragraph pointer whenever the current sentence is no longer found in
    the current paragraph's text but IS found in the next one. A single-paragraph essay
    (no blank lines) degrades gracefully - everything maps to paragraph 0.
    """
    indices = []
    p = 0
    for s in sentences:
        while p < len(paragraphs) - 1 and s not in paragraphs[p] and s in paragraphs[p + 1]:
            p += 1
        indices.append(p)
    return indices


# The whole pipeline, start to finish, for one piece of submitted text: segment into
# sentences and paragraphs, extract all 7 raw features per sentence, z-score them,
# run both trained models, build the per-factor "why" explanations, roll up into
# essay-level and paragraph-level scores, and compute the paragraph-outlier diagnostic.
# This is the ONLY function backend/main.py's POST /analyze calls - see this file's
# module docstring for the guarantee that every step here reuses Phase 2's own
# scripts/ code rather than reimplementing it.
def analyze_essay(text: str) -> dict:
    state = get_state()
    sentences = _clean_sentences(text.strip())
    if not sentences:
        raise ValueError("No valid sentences found after segmentation/filtering.")

    # --- passage-level grouping: which paragraph (blank-line-separated block in the
    # original submission) each sentence belongs to. Detection still runs at the sentence
    # level (unchanged) - this only labels sentences for the paragraph-level rollup below,
    # so a reader can see "this whole paragraph reads as AI-associated" even when no single
    # sentence in it is individually extreme. ---
    paragraph_texts = _split_into_paragraphs(text)
    paragraph_indices = _assign_paragraphs(sentences, paragraph_texts)

    # --- per-sentence perplexity + logprob variance ---
    perplexities, logprob_vars = [], []
    for idx, s in enumerate(sentences):
        start = max(0, idx - F.MAX_CONTEXT_SENTENCES)
        context = " ".join(sentences[start:idx])
        token_lps = F.sentence_token_logprobs(s, context)
        perplexities.append(F.sentence_perplexity(token_lps))
        logprob_vars.append(F.token_logprob_variance(token_lps))

    # --- essay-level burstiness + n-gram repetition (constant across all sentences) ---
    burst = F.essay_burstiness(sentences, perplexities)
    ngram = F.essay_ngram_repetition(sentences)

    baseline = state.baseline
    sentence_results = []
    essay_probas = []

    for idx, s in enumerate(sentences):
        raw = {
            "perplexity": perplexities[idx],
            "logprob_variance": logprob_vars[idx],
            "sentence_length_variance": burst["sentence_length_variance"],
            "syntactic_depth_variance": burst["syntactic_depth_variance"],
            "perplexity_variance": burst["perplexity_variance"],
            "ngram_repeat_rate": ngram["ngram_repeat_rate"],
            "transition_repetition_rate": ngram["transition_repetition_rate"],
        }
        z = {f"{k}_z": _zscore(v, k, baseline) for k, v in raw.items()}

        # --- essay-level model (per-sentence probability, averaged later) ---
        essay_score = None
        if all(z[f"{f}_z" if not f.endswith("_z") else f] is not None for f in
               [f.replace("_z", "") for f in ESSAY_MODEL_FEATURES]):
            x_essay = [[z[f] for f in ESSAY_MODEL_FEATURES]]
            essay_score = float(state.essay_model.predict_proba(x_essay)[0, 1])
            essay_probas.append(essay_score)

        # --- sentence-level model ---
        sentence_score = None
        if z["perplexity_z"] is not None and z["logprob_variance_z"] is not None:
            x_sent = [[z[f] for f in SENTENCE_MODEL_FEATURES]]
            sentence_score = float(state.sentence_model.predict_proba(x_sent)[0, 1])

        # --- all 7 essay-model features, each with a signed contribution (from the
        # essay-level model's coefficients, since it has the full 7-feature picture) ---
        essay_coefs = dict(zip(ESSAY_MODEL_FEATURES, state.essay_model.coef_[0]))
        contributions = []
        for f in ESSAY_MODEL_FEATURES:
            zval = z[f]
            if zval is None:
                continue
            signed = essay_coefs[f] * zval
            contributions.append({
                "feature": f,
                "value": round(zval, 4),
                "raw_value": round(raw[f.replace("_z", "")], 4),
                "contribution": round(abs(signed), 4),
                "direction": "toward_ai" if signed > 1e-9 else ("toward_human" if signed < -1e-9 else "neutral"),
            })
        contributions.sort(key=lambda c: -c["contribution"])
        top_features = contributions

        # --- the sentence-level model's own 2 raw features, shown in full (not just
        # "top 1-2") since there are only 2 to begin with - lets a reader verify the
        # sentence-level flag really is this thin, rather than taking the confidence
        # note's word for it ---
        sentence_coefs = dict(zip(SENTENCE_MODEL_FEATURES, state.sentence_model.coef_[0]))
        sentence_model_features = []
        for f in SENTENCE_MODEL_FEATURES:
            zval = z[f]
            if zval is None:
                continue
            signed = sentence_coefs[f] * zval
            sentence_model_features.append({
                "feature": f,
                "value": round(zval, 4),
                "raw_value": round(raw[f.replace("_z", "")], 4),
                "contribution": round(abs(signed), 4),
                "direction": "toward_ai" if signed > 1e-9 else ("toward_human" if signed < -1e-9 else "neutral"),
            })

        sentence_results.append({
            "index": idx,
            "text": s,
            "paragraph_index": paragraph_indices[idx],
            "sentence_level_score": sentence_score,
            "confidence": "low",
            "confidence_note": SENTENCE_CONFIDENCE_NOTE,
            "sentence_model_features": sentence_model_features,
            "top_features": top_features,
            "essay_model_local_score": essay_score,
            "essay_model_local_score_note": ESSAY_LOCAL_SCORE_NOTE,
        })

    essay_ai_likelihood = sum(essay_probas) / len(essay_probas) if essay_probas else None

    # --- passage-level rollup: mean essay_model_local_score per paragraph, the same
    # averaging pattern used for the essay-level score above, just scoped to one
    # paragraph's sentences instead of all of them. ---
    paragraph_results = []
    num_paragraphs = max(paragraph_indices) + 1 if paragraph_indices else 0
    for p in range(num_paragraphs):
        p_scores = [
            r["essay_model_local_score"] for r in sentence_results
            if r["paragraph_index"] == p and r["essay_model_local_score"] is not None
        ]
        p_sentence_count = sum(1 for r in sentence_results if r["paragraph_index"] == p)
        paragraph_results.append({
            "index": p,
            "sentence_count": p_sentence_count,
            "ai_likelihood_score": round(sum(p_scores) / len(p_scores), 4) if p_scores else None,
        })

    # --- exploratory paragraph-outlier diagnostic: how different is each paragraph's
    # OWN writing from the other paragraphs in this same essay - raw statistics only,
    # never run through either trained model or compared against the global baseline.
    # See PHASE2_PROCESS.md Step 7.5: a similar idea (sentence-vs-neighborhood deviation)
    # was already tried as a TRAINED feature and found to point the wrong way (human
    # sentences deviated from their neighbors MORE than AI-edited ones). That finding is
    # about a classifier input; this is shown raw as a magnitude signal only, never an
    # AI/human claim - hence "statistically different," not "looks AI." Needs >= 2
    # paragraphs to have anything to compare against. ---
    NOTABLE_DEVIATION = 1.5
    paragraph_outlier_note = None
    if num_paragraphs >= 2:
        paragraph_sentence_lists = [[] for _ in range(num_paragraphs)]
        paragraph_perplexity_lists = [[] for _ in range(num_paragraphs)]
        for idx, s in enumerate(sentences):
            p = paragraph_indices[idx]
            paragraph_sentence_lists[p].append(s)
            paragraph_perplexity_lists[p].append(perplexities[idx])

        p_mean_perplexity = []
        for plist in paragraph_perplexity_lists:
            valid = [v for v in plist if v is not None]
            p_mean_perplexity.append(sum(valid) / len(valid) if valid else None)
        # Variance over a single sentence is trivially 0.0, not a genuine "uniform
        # paragraph" finding - treat as unavailable (None) so it's excluded from the
        # comparison rather than distorting it with a degenerate zero.
        p_length_variance = [
            F.sentence_length_variance(slist) if len(slist) >= 2 else None
            for slist in paragraph_sentence_lists
        ]
        p_depth_variance = [
            F.syntactic_depth_variance(slist) if len(slist) >= 2 else None
            for slist in paragraph_sentence_lists
        ]

        perplexity_devs = F.paragraph_relative_deviation(p_mean_perplexity)
        length_devs = F.paragraph_relative_deviation(p_length_variance)
        depth_devs = F.paragraph_relative_deviation(p_depth_variance)

        for p in range(num_paragraphs):
            devs = (perplexity_devs[p], length_devs[p], depth_devs[p])
            notable = any(d is not None and abs(d) > NOTABLE_DEVIATION for d in devs)
            paragraph_results[p]["local_outlier"] = {
                "mean_perplexity_deviation": round(devs[0], 4) if devs[0] is not None else None,
                "sentence_length_variance_deviation": round(devs[1], 4) if devs[1] is not None else None,
                "syntactic_depth_variance_deviation": round(devs[2], 4) if devs[2] is not None else None,
                "notable": notable,
            }

        paragraph_outlier_note = (
            "Exploratory, not model-validated: compares each paragraph's own writing "
            "statistics only against the OTHER paragraphs in this same essay (not the "
            "global baseline, not run through either trained model). A large deviation "
            "means this paragraph is statistically different from the rest of the "
            "essay - which could reflect a style shift, a topic change, AI editing, or "
            "just normal writing variation. This project's own evaluation data found "
            "that sentences blending in smoothly with their neighbors were, if "
            "anything, slightly MORE associated with AI-edited content than sentences "
            "that stand out - so a flagged paragraph should not be read as 'this is the "
            "AI part.'"
        )

    return {
        "essay_level": {
            "ai_likelihood_score": round(essay_ai_likelihood, 4) if essay_ai_likelihood is not None else None,
            "confidence_note": ESSAY_CONFIDENCE_NOTE,
        },
        "sentences": sentence_results,
        "paragraphs": paragraph_results,
        "paragraph_outlier_note": paragraph_outlier_note,
        "fairness_note": FAIRNESS_NOTE,
        "sentence_count": len(sentences),
    }
