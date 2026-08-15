"""
Phase 2, Step 2: four independent, pure feature-extraction functions.

GPT-2 (base, loaded once via `get_model()`) is used purely as a measurement instrument -
it never outputs a verdict. All functions here take text in and return numbers out.

Design note on Functions 1 & 2 (perplexity, token log-prob variance): both are derived
from the SAME underlying per-token log-probabilities, computed once by
`sentence_token_logprobs()`. This guarantees a sentence is only ever scored by one GPT-2
forward pass, even though perplexity and variance are exposed as separate, independently
callable pure functions.

Context-scoring approach: rather than concatenating context+sentence into one string and
masking context token labels to -100, we tokenize the context and the sentence
SEPARATELY and concatenate their token-id sequences directly. This sidesteps any
ambiguity about exactly which token span belongs to the sentence after BPE merging across
the context/sentence boundary - the split point is known exactly, by construction.
"""
import statistics
from collections import Counter

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from text_utils import _get_nlp, split_sentences

MODEL_NAME = "gpt2"
MAX_CONTEXT_SENTENCES = 3

_model = None
_tokenizer = None


def get_model():
    """Loads GPT-2 once and caches it. Never called for anything but measurement."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def sentence_token_logprobs(sentence_text: str, preceding_context: str = ""):
    """
    Runs one GPT-2 forward pass over `preceding_context + sentence_text` and returns the
    list of per-token log-probabilities for ONLY the sentence's own tokens (context
    tokens are never scored). This is the single forward pass shared by Functions 1 & 2.

    If preceding_context is empty, the sentence's own first token also can't be scored
    (no left context at all for it) - it's skipped, same convention used in the Step 0
    sanity check. Returns an empty list if the sentence yields no scorable tokens (e.g. a
    single-token sentence with no preceding context).
    """
    model, tokenizer = get_model()

    context_ids = tokenizer(preceding_context, return_tensors="pt")["input_ids"] if preceding_context else None
    sentence_ids = tokenizer(sentence_text, return_tensors="pt")["input_ids"]

    if context_ids is not None and context_ids.shape[1] > 0:
        combined_ids = torch.cat([context_ids, sentence_ids], dim=1)
        context_len = context_ids.shape[1]
    else:
        combined_ids = sentence_ids
        context_len = 0

    with torch.no_grad():
        logits = model(combined_ids).logits  # (1, seq_len, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)

    total_len = combined_ids.shape[1]
    token_logprobs = []
    start = max(context_len, 1)  # position 0 can never be scored (no left context)
    for j in range(start, total_len):
        actual_token_id = combined_ids[0, j].item()
        lp = log_probs[0, j - 1, actual_token_id].item()
        token_logprobs.append(lp)

    return token_logprobs


# =============================================================================
# ALTERNATIVE MODEL BACKEND (INACTIVE) - Hugging Face Inference API
# =============================================================================
# Everything below is commented out and NOT executed anywhere - get_model() and
# sentence_token_logprobs() above (the local `transformers` + torch implementation) are
# the only active code path. This block is a reference implementation of the same
# function using the Hugging Face hosted Inference API instead of a locally-loaded
# model, kept here so it's easy to find and switch to deliberately - see README.md's
# "Model backend: local GPT-2 vs. Hugging Face Inference API" section for why you might
# want to, and how to actually enable it (uncomment this block, add `import os` to this
# file's imports, install huggingface_hub, set the HF_API_TOKEN environment variable,
# and swap the call site in this file from sentence_token_logprobs() to
# sentence_token_logprobs_hf_api()).
#
# Same base "gpt2" checkpoint either way - not a different or stronger model, just a
# different place it runs. That means it SHOULD produce equivalent perplexity/log-prob
# numbers to the local path, but that equivalence has not been verified end-to-end
# (tokenization and floating-point details can differ subtly between a local forward
# pass and a hosted inference server) - treat this as a documented option, not a
# validated one, until someone actually runs the comparison.
#
# from huggingface_hub import InferenceClient
#
# HF_API_MODEL = "gpt2"
# HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # required - free at huggingface.co/settings/tokens
#
# _hf_client = None
#
#
# def get_hf_client():
#     """API equivalent of get_model() above - no weights to load locally, just a
#     lazily-constructed client reused across calls."""
#     global _hf_client
#     if _hf_client is None:
#         if not HF_API_TOKEN:
#             raise RuntimeError("HF_API_TOKEN environment variable not set")
#         _hf_client = InferenceClient(model=HF_API_MODEL, token=HF_API_TOKEN)
#     return _hf_client
#
#
# def sentence_token_logprobs_hf_api(sentence_text: str, preceding_context: str = ""):
#     """
#     API equivalent of sentence_token_logprobs() above. Uses decoder_input_details=True
#     (Hugging Face's Text Generation Inference "prefill" mode), which returns a logprob
#     for every INPUT token, not just newly-generated ones - the same teacher-forced
#     scoring the local version does via one forward pass over the full input.
#
#     Caveat vs. the local version: the local implementation tokenizes context and
#     sentence SEPARATELY so the exact token boundary between them is known by
#     construction (see this file's module docstring). The API only tokenizes the
#     combined string once, so the context/sentence split has to be re-derived from the
#     combined token count - this is a real, not-yet-handled gap in this sketch, not a
#     solved equivalent. One real network round-trip per sentence, unlike the local path's
#     one in-process forward pass - expect meaningfully higher latency at essay scale.
#     """
#     client = get_hf_client()
#     full_text = f"{preceding_context} {sentence_text}".strip() if preceding_context else sentence_text
#
#     result = client.text_generation(
#         full_text,
#         max_new_tokens=1,
#         details=True,
#         decoder_input_details=True,
#     )
#
#     # One entry per input token, in order: {id, text, logprob}. The very first token
#     # always has logprob=None (no left context for it, same convention as the local
#     # version's `start = max(context_len, 1)`).
#     prefill = result.details.prefill
#     return [t.logprob for t in prefill if t.logprob is not None]
# =============================================================================
# END ALTERNATIVE MODEL BACKEND
# =============================================================================


def sentence_perplexity(token_logprobs) -> float | None:
    """Function 1: exp(mean negative log-likelihood) over the sentence's own tokens."""
    if not token_logprobs:
        return None
    mean_neg_ll = -sum(token_logprobs) / len(token_logprobs)
    return torch.exp(torch.tensor(mean_neg_ll)).item()


def token_logprob_variance(token_logprobs) -> float | None:
    """Function 2: population variance of the sentence's own per-token log-probs."""
    if len(token_logprobs) < 2:
        return None
    return statistics.pvariance(token_logprobs)


# ---------------------------------------------------------------------------
# Function 3: essay-level burstiness
# ---------------------------------------------------------------------------

def sentence_length_variance(essay_sentences: list[str]) -> float:
    """Population variance of word count across all sentences in essay_sentences - one
    number per essay (or per paragraph, when called on just a paragraph's sentences from
    backend/pipeline.py's outlier diagnostic). 0.0 (not None) for fewer than 2 sentences,
    since variance over 0-1 values is trivially zero, not undefined."""
    lengths = [len(s.split()) for s in essay_sentences]
    return statistics.pvariance(lengths) if len(lengths) >= 2 else 0.0


def _parse_tree_depth(doc_sent) -> int:
    """Max depth from any token to the sentence root, walking token.head chains."""
    max_depth = 0
    for token in doc_sent:
        depth = 0
        cur = token
        while cur.head != cur and depth < 200:  # guard against pathological cycles
            depth += 1
            cur = cur.head
        max_depth = max(max_depth, depth)
    return max_depth


def syntactic_depth_variance(essay_sentences: list[str]) -> float:
    """Population variance of each sentence's max dependency-parse-tree depth
    (_parse_tree_depth() above) across essay_sentences - how much grammatical
    complexity swings from sentence to sentence. Same 0.0-for-under-2-sentences
    convention as sentence_length_variance()."""
    nlp = _get_nlp()
    if nlp is None:
        raise RuntimeError("spaCy model not available - syntactic_depth_variance requires en_core_web_sm")
    depths = [_parse_tree_depth(nlp(s)) for s in essay_sentences]
    return statistics.pvariance(depths) if len(depths) >= 2 else 0.0


def perplexity_variance(perplexities: list[float]) -> float:
    """Population variance of per-sentence perplexity across an essay (None entries
    excluded) - a THIRD burstiness sub-metric alongside length/depth variance, this one
    over GPT-2's own surprise level rather than a purely structural measurement."""
    valid = [p for p in perplexities if p is not None]
    return statistics.pvariance(valid) if len(valid) >= 2 else 0.0


def essay_burstiness(essay_sentences: list[str], perplexities: list[float]) -> dict:
    """Combines the three burstiness sub-metrics for one essay's sentence list."""
    return {
        "sentence_length_variance": sentence_length_variance(essay_sentences),
        "syntactic_depth_variance": syntactic_depth_variance(essay_sentences),
        "perplexity_variance": perplexity_variance(perplexities),
    }


# ---------------------------------------------------------------------------
# Step 7.5: sentence-local features (fixing the ai_edited blind spot)
#
# Every feature above except perplexity_z/logprob_variance_z is an ESSAY-WIDE constant
# joined onto every sentence in that essay - Step 7 showed this makes the model nearly
# blind to an AI-edited sentence sitting inside an otherwise-human hybrid essay, since
# the essay-wide signal (which looks human for hybrids) swamps everything else. These two
# features are deliberately LOCAL: computed relative to a sentence's own nearby
# neighborhood (or the rest of the essay excluding itself), never a single number
# repeated for the whole essay, and never the global human-corpus baseline (that
# normalization step happens later, same as every other feature, in the baseline/z-score
# step - these functions return raw deviations).
# ---------------------------------------------------------------------------

def local_perplexity_deviation(perplexities: list[float], idx: int) -> float | None:
    """
    Formula: (perplexities[idx] - mean(perplexities excluding idx)) / pstdev(perplexities
    excluding idx). Guards: returns None if fewer than 2 OTHER sentences have a valid
    (non-None) perplexity, or if their std is 0 (would make the ratio undefined/unstable).
    """
    others = [p for j, p in enumerate(perplexities) if j != idx and p is not None]
    if perplexities[idx] is None or len(others) < 2:
        return None
    mean = statistics.mean(others)
    std = statistics.pstdev(others)
    if std == 0:
        return None
    return (perplexities[idx] - mean) / std


def _local_window_deviation(values: list[float], idx: int, radius: int = 2) -> float | None:
    """
    Shared helper for local_length_deviation / local_depth_deviation. Formula:
    (values[idx] - mean(window excluding idx)) / pstdev(window excluding idx), where the
    window is [idx-radius, idx+radius] CLIPPED at the essay's boundaries (no padding) -
    a sentence near the start/end of an essay simply gets a smaller, asymmetric window
    (e.g. the first sentence's window is just the next 2 sentences, no left neighbors).
    Guards: returns None if fewer than 2 other sentences fall in the window, or their
    std is 0.
    """
    n = len(values)
    if values[idx] is None:
        return None
    window_idxs = [j for j in range(max(0, idx - radius), min(n, idx + radius + 1))
                   if j != idx and values[j] is not None]
    if len(window_idxs) < 2:
        return None
    window_vals = [values[j] for j in window_idxs]
    mean = statistics.mean(window_vals)
    std = statistics.pstdev(window_vals)
    if std == 0:
        return None
    return (values[idx] - mean) / std


def paragraph_relative_deviation(paragraph_values: list[float | None]) -> list[float | None]:
    """
    For each paragraph's own raw value, deviation from the mean of the OTHER paragraphs'
    values in the same essay: (value[i] - mean(others)) / pstdev(others). Same formula
    and same guards as _local_window_deviation() above, just applied paragraph-to-
    paragraph instead of sentence-to-neighbor-window - not a trained-model feature, used
    only for the backend's exploratory "this paragraph looks statistically different from
    the rest of this essay" diagnostic (see PHASE2_PROCESS.md Step 7.5 for why a similar
    idea was tried as a TRAINED feature and abandoned - that finding does not apply here
    since this is shown raw, never fed into either classifier).
    Guards: returns None for a paragraph if its own value is None, fewer than 2 other
    paragraphs have a valid value, or their pstdev is 0.
    """
    n = len(paragraph_values)
    result = []
    for i in range(n):
        if paragraph_values[i] is None:
            result.append(None)
            continue
        others = [v for j, v in enumerate(paragraph_values) if j != i and v is not None]
        if len(others) < 2:
            result.append(None)
            continue
        mean = statistics.mean(others)
        std = statistics.pstdev(others)
        if std == 0:
            result.append(None)
            continue
        result.append((paragraph_values[i] - mean) / std)
    return result


def local_burstiness_disruption(essay_sentences: list[str], radius: int = 2) -> dict:
    """
    Returns {"local_length_deviation": [...], "local_depth_deviation": [...]}, one value
    per sentence in essay_sentences, computed via _local_window_deviation() over word
    counts and spaCy dependency-parse-tree depths respectively.
    """
    nlp = _get_nlp()
    if nlp is None:
        raise RuntimeError("spaCy model not available - local_burstiness_disruption requires en_core_web_sm")
    lengths = [len(s.split()) for s in essay_sentences]
    depths = [_parse_tree_depth(nlp(s)) for s in essay_sentences]
    length_devs = [_local_window_deviation(lengths, i, radius) for i in range(len(essay_sentences))]
    depth_devs = [_local_window_deviation(depths, i, radius) for i in range(len(essay_sentences))]
    return {"local_length_deviation": length_devs, "local_depth_deviation": depth_devs}


# ---------------------------------------------------------------------------
# Function 4: n-gram repetition (whole-essay, redesigned)
# ---------------------------------------------------------------------------
#
# Design history (kept here rather than deleted, since it explains *why* the final
# design looks the way it does):
#   v1 - bigram/trigram type-token ratio over a fixed 4-sentence window. Pinned near a
#        1.0 ceiling (0.978-1.0) - too little text (15-25 words) for repeats to occur.
#   v2 - same TTR idea, widened to a ~120-word window. Barely moved (0.975-1.0).
#   v3 - v2 + stripped stopwords before building n-grams. Made it WORSE - trigram_ttr
#        went completely flat at 1.0 for every window in both a human and an ai spot-test
#        essay, because fewer tokens per window (content words only) made 3-word
#        sequences even more likely to be unique by chance.
#   v4 (current) - abandoned local-window TTR entirely. Any grammatical English text
#        has near-unique local 2-3 word sequences almost by default - that's just how
#        combinatorially rich language is, regardless of who wrote it. The plausible
#        AI-tell is a phrase or transition recurring ACROSS separate parts of one essay
#        (e.g. reusing "in today's society" three times), which a local window can never
#        see by construction. So v4 counts exact bigram/trigram repeats across the WHOLE
#        essay and reports a repeat RATE (not a TTR) - joined onto every sentence row,
#        the same pattern Function 3's burstiness already uses.

TRANSITION_WORDS = [
    "however", "moreover", "furthermore", "in conclusion", "additionally",
    "therefore", "nevertheless", "nonetheless", "consequently", "in addition",
    "on the other hand", "as a result", "in fact", "for example", "for instance",
    "in other words", "ultimately", "overall", "in summary", "thus",
]


def _content_word_tokens(sentence: str) -> list[str]:
    """Lowercased content words only (alphabetic, non-stopword) via spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        raise RuntimeError("spaCy model not available - n-gram repetition requires en_core_web_sm")
    doc = nlp(sentence)
    return [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]


def _ngram_repeat_rate(content_words: list[str], n: int):
    """
    Fraction of n-gram occurrences in the essay that are repeats of an earlier
    occurrence: (total_occurrences - distinct_ngrams) / total_occurrences. 0.0 means no
    repeated n-grams at all; higher values mean more phrase reuse across the essay. Also
    returns the single most-repeated n-gram and its count, for interpretability.
    """
    if len(content_words) < n:
        return None, None, 0
    grams = [tuple(content_words[i:i + n]) for i in range(len(content_words) - n + 1)]
    total = len(grams)
    if total == 0:
        return None, None, 0
    counts = Counter(grams)
    distinct = len(counts)
    rate = (total - distinct) / total
    top_gram, top_count = counts.most_common(1)[0]
    return rate, top_gram, top_count


def essay_ngram_repetition(essay_sentences: list[str]) -> dict:
    """
    Function 4 (redesigned, whole-essay scope): bigram/trigram repeat rate over content
    words, plus transition-word repetition rate over the full original text. Computed
    once per essay; callers join the result onto every sentence row for that essay
    (same pattern as essay_burstiness()).
    """
    content_words = []
    for s in essay_sentences:
        content_words.extend(_content_word_tokens(s))

    bigram_rate, top_bigram, top_bigram_count = _ngram_repeat_rate(content_words, 2)
    trigram_rate, top_trigram, top_trigram_count = _ngram_repeat_rate(content_words, 3)
    valid_rates = [r for r in (bigram_rate, trigram_rate) if r is not None]
    ngram_repeat_rate = sum(valid_rates) / len(valid_rates) if valid_rates else None

    full_text_lower = " ".join(essay_sentences).lower()
    tw_counts = {tw: full_text_lower.count(tw) for tw in TRANSITION_WORDS}
    most_repeated_word, most_repeated_count = max(tw_counts.items(), key=lambda kv: kv[1])
    n_sentences = len(essay_sentences)
    transition_repetition_rate = most_repeated_count / n_sentences if n_sentences else 0.0

    return {
        "bigram_repeat_rate": bigram_rate,
        "trigram_repeat_rate": trigram_rate,
        "ngram_repeat_rate": ngram_repeat_rate,
        "top_repeated_bigram": " ".join(top_bigram) if top_bigram and top_bigram_count >= 2 else None,
        "top_repeated_bigram_count": top_bigram_count,
        "top_repeated_trigram": " ".join(top_trigram) if top_trigram and top_trigram_count >= 2 else None,
        "top_repeated_trigram_count": top_trigram_count,
        "most_repeated_transition": most_repeated_word if most_repeated_count > 0 else None,
        "transition_repetition_rate": transition_repetition_rate,
    }
